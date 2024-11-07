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

from recognition import OnlineASR
from generation import VoiceGenerator
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
        _, message = self._update_conversation_context(ResponseLLMAgent.greeting_message())
        self.voice_generator.generate_async(text=message["content"], id=message["id"])

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
            self._maybe_respond()

    def _maybe_respond(self):
        need_response = self.conversation_context.messages and self.conversation_context.messages[-1]["role"] == "user"
        if need_response:
            self.conversation_context.process_interrupted_messages()
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

    def on_assistant_audio_end(self, speech_id, duration):
        messages = self.conversation_context.get_messages(filter=lambda msg: msg["role"] == "assistant" and msg["id"] == speech_id)
        assert messages, f"Message with speech_id {speech_id} not found"
        self.conversation_context.update_message(messages[0]["id"], audio_duration=duration)

    def on_user_interrupt(self, speech_id, interrupted_at):
        messages = self.conversation_context.get_messages(filter=lambda msg: msg["role"] == "assistant" and msg["id"] == speech_id)
        assert messages, f"Message with speech_id {speech_id} not found"
        self.conversation_context.update_message(messages[0]["id"], interrupted_at=interrupted_at)

    def shutdown(self):
        self.running = False
        # self._transcription_changed.set()
        # self._response_agent_loop.join(timeout=5)


async def handle_connection(websocket):
    log_dir = f"logs/conversations/{get_timestamp()}"
    logger.add(f"{log_dir}/server.log", rotation="100 MB")

    logger.info("New connection is started")

    last_assistant_speech_id = 0

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
    async def handle_generated_audio(audio_chunk, speech_id, duration=None):
        # audio_chunk is f32le
        nonlocal last_assistant_speech_id
        if audio_chunk is None: # end of audio
            audio_output_stream.flush()
            conversation.on_assistant_audio_end(speech_id, duration)
        else:
            audio_output_stream.put(audio_chunk)
            audio_output_saver.write(convert_f32le_to_s16le(audio_chunk))
            last_assistant_speech_id = speech_id # hack

    @logger.catch
    async def handle_webm_audio(audio_chunk):
        # chunk is webm
        nonlocal last_assistant_speech_id
        if audio_chunk is not None and websocket.open:
            metadata = {
                "type": "audio",
                # FIXME: we can't get audio id here because we're using ffmpeg to convert source audio to webm and lose audio id
                "speech_id": last_assistant_speech_id
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
            else:
                try:
                    message_data = json.loads(message)
                    message_type = message_data["type"]
                    if message_type == "start_recording":
                        logger.info("Received start_recording message")
                        audio_input_stream.start()
                    elif message_type == "stop_recording":
                        logger.info("Received stop_recording message")
                        audio_input_stream.stop()
                    elif message_type == "interrupt":
                        logger.info("Received interrupt message")
                        conversation.on_user_interrupt(
                            speech_id=message_data["speech_id"], 
                            interrupted_at=message_data["interrupted_at_ms"] / 1000.0)
                    else:
                        logger.warning(f"Received unexpected type: {message_type}")
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON message: {message}")
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
