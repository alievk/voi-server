import os
import asyncio
import json
import threading
from collections import deque
from datetime import datetime
import signal
from typing import Callable

import torch
import numpy as np
import websockets
from loguru import logger

import litellm

from recognition import OnlineASR
from generation import VoiceGenerator, DummyVoiceGenerator
from audio import AudioOutputStream, AudioInputStream, WavSaver
from llm import get_agent_config, stringify_content, ConversationContext, BaseLLMAgent, CharacterLLMAgent, CharacterEchoAgent


class Conversation:
    def __init__(
        self, 
        asr: OnlineASR, 
        voice_generator: VoiceGenerator,
        character_agent: BaseLLMAgent,
        conversation_context: ConversationContext=None
    ):
        self.asr = asr
        self.voice_generator = voice_generator
        self.character_agent = character_agent
        self.conversation_context = conversation_context

        self._event_loop = asyncio.get_event_loop()

    def greeting(self):
        message = self.character_agent.greeting_message()
        message = self.conversation_context.add_message(message)
        self._generate_voice(message)

    @logger.catch
    async def handle_input_audio(self, audio_chunk):
        """ audio_chunk is f32le """
        end_of_audio = audio_chunk is None
        asr_result = self.asr.process_chunk(audio_chunk, finalize=end_of_audio)

        if asr_result and (asr_result["confirmed_text"] or asr_result["unconfirmed_text"]):
            message = self._create_message_from_transcription(asr_result)
            messages = self.conversation_context.get_messages()
            if messages and messages[-1]["role"] == "user":
                self.conversation_context.update_message(messages[-1]["id"], content=message["content"], handled=False)
            else:
                self.conversation_context.add_message(message)

        if end_of_audio:
            self._maybe_respond()

    def on_manual_text(self, text):
        message = {
            "role": "user",
            "content": text,
            "time": datetime.now()
        }
        self.conversation_context.add_message(message)
        self._maybe_respond()

    def _maybe_respond(self):
        need_response = self.conversation_context.messages and self.conversation_context.messages[-1]["role"] == "user"
        if need_response:
            self.conversation_context.process_interrupted_messages()
            response = self.character_agent.completion(self.conversation_context)

            message = {
                "role": "assistant",
                "content": response,
                "time": datetime.now()
            }
            new_message = self.conversation_context.add_message(message)

            self._generate_voice(new_message)

    def _generate_voice(self, message):
        if message.get("file"): # play cached audio
            content = f"file:{message['file']}"
        elif isinstance(message["content"], dict):
            self.voice_generator.maybe_set_voice_tone(message["content"].get("voice_tone"))
            content = message["content"]["text"]
        else:
            content = message["content"]

        self.voice_generator.generate_async(text=content, id=message["id"])

    def _create_message_from_transcription(self, transcription):
        confirmed = transcription["confirmed_text"]
        unconfirmed = transcription["unconfirmed_text"]

        text = BaseLLMAgent.format_transcription(confirmed, unconfirmed)
        
        return {
            "role": "user",
            "content": text,
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


async def handle_connection(websocket):
    log_dir = f"logs/conversations/{get_timestamp()}"
    logger.add(f"{log_dir}/server.log", rotation="100 MB")

    logger.info("New connection is started")

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
            audio_input_saver.write(audio_chunk)

    @logger.catch
    async def handle_context_changed(context):
        messages = context.get_messages(filter=lambda msg: not msg["handled"])
        for msg in messages:
            data = {
                "type": "message",
                "role": msg["role"],
                "content": stringify_content(msg["content"]),
                "time": msg["time"].strftime("%H:%M:%S"),
                "id": msg["id"]
            }
            logger.info("Sending message: {}", data)
            try:
                await websocket.send(serialize_message(data))
                context.update_message(msg["id"], handled=True)
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    @logger.catch
    async def handle_generated_audio(audio_chunk, speech_id, duration=None):
        # audio_chunk is f32le
        nonlocal last_assistant_speech_id
        if audio_chunk is None: # end of audio
            audio_output_stream.flush()
            conversation.on_assistant_audio_end(speech_id, duration)
        else:
            audio_output_stream.put(audio_chunk)
            audio_output_saver.write(audio_chunk)
            last_assistant_speech_id = speech_id # hack

    @logger.catch
    async def handle_webm_audio(audio_chunk):
        # chunk is webm
        nonlocal last_assistant_speech_id
        if audio_chunk is not None:
            metadata = {
                "type": "audio",
                # FIXME: we can't get audio id here because we're using ffmpeg to convert source audio to webm and lose audio id
                "speech_id": last_assistant_speech_id
            }
            try:
                await websocket.send(serialize_message(metadata, audio_chunk))
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")

    @logger.catch
    async def read_init_message(websocket):
        data = await websocket.recv()
        init_message = json.loads(data)
        assert init_message["type"] == "init", f"The first message must be a JSON with 'init' type, but got: {init_message}"
        assert "agent_name" in init_message, f"The first message must contain 'agent_name' field, but got: {init_message}"
        return init_message
    
    init_message = await read_init_message(websocket)
    agent_config = get_agent_config(init_message["agent_name"])

    logger.info("Initializing speech recognition")
    asr = OnlineASR(
        cached=True
    )

    logger.info("Initializing voice generation")
    if agent_config["tts_model"] == "dummy":
        voice_generator = DummyVoiceGenerator()
    else:
        voice_generator = VoiceGenerator(
            cached=True,
            model_name=agent_config["tts_model"],
            generated_audio_cb=handle_generated_audio,
            mute_narrator=agent_config.get("mute_narrator", False)
        )
    voice_generator.maybe_set_voice_tone(agent_config.get("narrator_voice_tone"), role="narrator")
    voice_generator.start()

    logger.info("Initializing audio input stream")
    audio_input_stream = AudioInputStream(
        handle_input_audio,
        chunk_size_ms=1000,
        input_sample_rate=16000, # client sends 16000
        output_sample_rate=asr.sample_rate
    )
    audio_input_stream.start()

    logger.info("Initializing audio output stream")
    audio_output_stream = AudioOutputStream(
        handle_webm_audio,
        input_sample_rate=voice_generator.sample_rate,
        output_sample_rate=voice_generator.sample_rate
    )
    audio_output_stream.start()

    logger.info("Initializing response agent")
    if agent_config["llm_model"] == "echo":
        character_agent = CharacterEchoAgent()
    else:
        character_agent = CharacterLLMAgent.from_config(agent_config)

    conversation_context = ConversationContext(
        context_changed_cb=handle_context_changed
    )
    
    logger.info("Initializing conversation")
    conversation = Conversation(
        asr=asr,
        voice_generator=voice_generator,
        character_agent=character_agent,
        conversation_context=conversation_context
    )

    logger.info("Initializing audio input saver")
    audio_input_saver = WavSaver(
        f"{log_dir}/incoming.wav", 
        sample_rate=audio_input_stream.output_sample_rate
    )

    logger.info("Initializing audio output saver")
    audio_output_saver = WavSaver(
        f"{log_dir}/outgoing.wav", 
        sample_rate=voice_generator.sample_rate
    )

    # we need to send speech id to the client to identify which agent speech was interrupted by the user
    # this variable is a hack to keep track of the last speech id because ffmpeg audio converter loses speech id
    # one possible solution is to use a converter which will preserve metadata like speech id
    last_assistant_speech_id = 0

    conversation.greeting()

    logger.info("Entering main loop")
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
                    elif message_type == "manual_text":
                        logger.info("Received manual_text message")
                        conversation.on_manual_text(message_data["content"])
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

        torch.cuda.empty_cache()
        
    logger.info("Connection is done")


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
    )
    logger.info("WebSocket server started on wss://0.0.0.0:8765")

    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        logger.info("Server is shutting down...")
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    asyncio.run(main())
