import asyncio
import ssl
import json
import threading
from collections import deque
from datetime import datetime
import signal

import numpy as np
import torch
import websockets
from loguru import logger

import litellm

from stt import OnlineASR
from tts import VoiceGenerator
from audio import AudioOutputStream, AudioInputStream, WavSaver
from llm import ConversationContext, BaseLLMAgent, ResponseLLMAgent


class Conversation:
    def __init__(self, asr, voice_generator, context_changed_cb=None):
        self.asr = asr
        self.voice_generator = voice_generator
        self.context_changed_cb = context_changed_cb
        self.conversation_context = ConversationContext()

        self._response_agent = ResponseLLMAgent()
        self._transcription_changed = threading.Event()
        self._event_loop = asyncio.get_event_loop()
        self._response_agent_loop = threading.Thread(target=self._response_agent_loop, name='response-agent-loop', daemon=True)
        # self._response_agent_loop.start()
        self.running = True
        self.last_agent_message = None

    def greeting(self):
        message = ResponseLLMAgent.greeting_message()
        self._update_conversation_context(message)
        self.voice_generator.generate_async(text=message["content"])

    @logger.catch
    async def handle_input_audio(self, audio_chunk):
        """ audio_chunk is f32le """
        end_of_audio = audio_chunk is None
        asr_result = self.asr.process_chunk(audio_chunk, finalize=end_of_audio)

        if asr_result and (asr_result["confirmed_text"] or asr_result["unconfirmed_text"]):
            message = self._create_message_from_transcription(asr_result)
            context_changed, changed_message = self._update_conversation_context(message)
            # if context_changed:
            #     self._transcription_changed.set()

        if end_of_audio:
            need_response = self.conversation_context.messages and self.conversation_context.messages[-1]["role"] == "user"
            if need_response:
                response = self._response_agent.completion(self.conversation_context)
                agent_message = {
                    "role": "assistant",
                    "content": response["content"],
                    "time": datetime.now()
                }
                _, new_message = self._update_conversation_context(agent_message)
                self.voice_generator.generate_async(text=response["content"], id=new_message["id"])

    def _create_message_from_transcription(self, transcription):
        confirmed = transcription["confirmed_text"]
        unconfirmed = transcription["unconfirmed_text"]

        text = BaseLLMAgent.format_transcription(confirmed, unconfirmed)
        
        return {
            "role": "user",
            "content": text,
            "time": datetime.now()
        }

    def _update_conversation_context(self, message):
        compare_ignore_case = lambda x, y: x.strip().lower() == y.strip().lower()
        context_changed, changed_message = self.conversation_context.add_message(message, text_compare_f=compare_ignore_case)
        if context_changed:
            asyncio.run_coroutine_threadsafe(self.context_changed_cb(self.conversation_context), self._event_loop)
        return context_changed, changed_message

    def _response_agent_loop(self):
        while True:
            self._transcription_changed.wait()
            self._transcription_changed.clear()

            if not self.running:
                break
            
            logger.error("Context changed:\n{}", self.conversation_context.to_text())
            response = self._response_agent.completion(self.conversation_context)
            logger.error("Agent response:\n{}", response["content"])

            self.last_agent_message = {
                "role": "assistant",
                "content": response["content"],
                "time": datetime.now()
            }

    def shutdown(self):
        self.running = False
        # self._transcription_changed.set()
        # self._response_agent_loop.join(timeout=5)


async def handle_connection(websocket):
    log_dir = f"logs/conversations/{get_timestamp()}"
    logger.add(f"{log_dir}/server.log", rotation="100 MB")

    logger.info("New connection is started")

    def convert_f32le_to_s16le(buffer):
        return (buffer * 32768.0).astype(np.int16).tobytes()

    def serialize_message(metadata, blob=None):
        metadata = json.dumps(metadata).encode('utf-8')
        metadata_length = len(metadata).to_bytes(4, byteorder='big')
        if blob is None:
            return metadata_length + metadata
        else:
            return metadata_length + metadata + blob

    @logger.catch
    async def handle_input_audio(audio_chunk):
        # audio_chunk is f32le
        await conversation.handle_input_audio(audio_chunk)
        if audio_chunk is not None:
            audio_input_saver.write(convert_f32le_to_s16le(audio_chunk))

    @logger.catch
    async def handle_context_changed(context):
        messages = context.get_messages(filter=lambda msg: not msg["handled"])
        for msg in messages:
            data = {
                "type": "message",
                "role": msg["role"],
                "content": msg["content"],
                "time": msg["time"].strftime("%H:%M:%S"),
                "id": msg["id"]
            }
            logger.info("Sending message: {}", data)
            await websocket.send(serialize_message(data))
            context.update_message(msg["id"], handled=True)

    @logger.catch
    async def handle_generated_audio(audio_chunk, id=None):
        # audio_chunk is f32le
        if audio_chunk is None: # end of audio
            audio_output_stream.flush()
        else:
            audio_output_stream.put(audio_chunk, id=id)
            audio_output_saver.write(convert_f32le_to_s16le(audio_chunk))

    @logger.catch
    async def handle_webm_audio(audio_chunk, id=None):
        # chunk is webm
        if audio_chunk is not None and websocket.open:
            metadata = {
                "type": "audio",
                "id": 0 if id is None else id
            }
            await websocket.send(serialize_message(metadata, audio_chunk))

    asr = OnlineASR(
        cached=True
    )

    voice_generator = VoiceGenerator(
        cached=True,
        generated_audio_cb=handle_generated_audio
    )

    audio_input_stream = AudioInputStream(
        handle_input_audio,
        chunk_size_ms=1000,
        input_sample_rate=16000, # client sends 16000
        output_sample_rate=asr.sample_rate
    )
    
    audio_output_stream = AudioOutputStream(
        handle_webm_audio,
        input_sample_rate=voice_generator.sample_rate,
        output_sample_rate=voice_generator.sample_rate
    )
    
    conversation = Conversation(
        asr=asr,
        voice_generator=voice_generator,
        context_changed_cb=handle_context_changed
    )

    audio_input_saver = WavSaver(
        f"{log_dir}/incoming.wav", 
        sample_rate=audio_input_stream.output_sample_rate
    )

    audio_output_saver = WavSaver(
        f"{log_dir}/outgoing.wav", 
        sample_rate=voice_generator.sample_rate
    )

    voice_generator.start()
    audio_input_stream.start()
    audio_output_stream.start()

    conversation.greeting()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio_input_stream.put(message)
            elif message == 'START_RECORDING':
                logger.info("Received START_RECORDING message")
                audio_input_stream.start()
            elif message == 'STOP_RECORDING':
                logger.info("Received STOP_RECORDING message")
                audio_input_stream.stop()
            else:
                logger.warning(f"Received unexpected message: {message}")
        logger.info("WebSocket connection closed, checking timeout")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed unexpectedly")
    finally:
        logger.info("Cleaning up resources")
        voice_generator.stop()
        audio_input_stream.stop()
        audio_output_stream.stop()
        audio_input_saver.close()
        audio_output_saver.close()
        conversation.shutdown()
        
    logger.info("Connection is done")


ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('localhost+2.pem', 'localhost+2-key.pem')

litellm.api_base = "http://13.43.85.180:4000"
litellm.api_key = "sk-1234"


def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


async def shutdown(signal, loop):
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def main():
    logger.add(f"logs/server/server-{get_timestamp()}.log", rotation="100 MB")

    loop = asyncio.get_running_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop))
        )

    server = await websockets.serve(
        handle_connection,
        "0.0.0.0",
        8765,
        ssl=ssl_context
    )
    logger.info("WebSocket server started on wss://0.0.0.0:8765")

    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        logger.info("Server is shutting down...")
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
