import asyncio
import multiprocessing
import os, sys
import ssl
import json
import subprocess
import threading
from collections import deque
from datetime import datetime
import signal
import wave

import numpy as np
import torch
import websockets
from loguru import logger

import litellm

from streaming import OnlineASR
from tts import VoiceGenerator
from prompts import response_agent_system_prompt, response_agent_greeting_message, response_agent_examples
from audio import AudioOutputStream


class WavSaver:
    def __init__(self, filename, channels=1, sample_width=2, sample_rate=16000, buffer_size=1024*1024):
        self.filename = filename
        self.channels = channels
        self.sample_width = sample_width
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.wav_file = None
        self.buffer = b''
        self._create_wav_file()

    def write(self, chunk):
        """ Input format is s16le """
        self.buffer += chunk
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()

    def _create_wav_file(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.wav_file = wave.open(os.path.join(self.filename), 'wb')
        self.wav_file.setnchannels(self.channels)
        self.wav_file.setsampwidth(self.sample_width)
        self.wav_file.setframerate(self.sample_rate)

    def _flush_buffer(self):
        if self.wav_file and self.buffer:
            self.wav_file.writeframes(self.buffer)
            self.buffer = b''

    def close(self):
        self._flush_buffer()
        if self.wav_file:
            self.wav_file.close()


class AudioInputStream:
    def __init__(self, converted_audio_cb, chunk_size_ms=1000, input_sample_rate=16000, output_sample_rate=16000):
        """ converted_audio_cb receives f32le audio chunks """
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.ffmpeg_process = None
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.converted_audio_cb = converted_audio_cb
        self.chunk_size = int(chunk_size_ms / 1000 * self.output_sample_rate * 2)
        self.running = False
        self.threads = []

    def start(self):
        if self.running:
            return

        self.running = True
        self.ffmpeg_process = self._start_ffmpeg_process()
        self.threads = [
            threading.Thread(target=self._ffmpeg_in_pipe, name='ffmpeg-in-pipe', daemon=True),
            threading.Thread(target=self._ffmpeg_out_pipe, name='ffmpeg-out-pipe', daemon=True),
            threading.Thread(target=self._audio_callback, name='ffmpeg-callback', daemon=True, args=(asyncio.get_event_loop(),))
        ]
        for thread in self.threads:
            thread.start()

    def put(self, chunk):
        """ chunk is webm """
        if not self.running:
            logger.warning("Audio input stream is not running, skipping chunk")
            return

        self.input_queue.put(chunk)

    def _ffmpeg_in_pipe(self):
        while self.running:
            chunk = self.input_queue.get()
            if chunk is None:
                break
            self.ffmpeg_process.stdin.write(chunk)
            self.ffmpeg_process.stdin.flush()

    def _ffmpeg_out_pipe(self):
        buffer = bytearray()
        while self.running:
            pcm_chunk = self.ffmpeg_process.stdout.read(1024)
            if not pcm_chunk:
                break
            
            buffer.extend(pcm_chunk)
            while len(buffer) >= self.chunk_size:
                self.output_queue.put(bytes(buffer[:self.chunk_size]))
                del buffer[:self.chunk_size]
        
        if buffer:
            self.output_queue.put(bytes(buffer))

        self.output_queue.put(None)

    def _audio_callback(self, loop):
        def convert_s16le_to_f32le(buffer):
            return np.frombuffer(buffer, dtype='<i2').flatten().astype(np.float32) / 32768.0

        while self.running:
            s16le_chunk = self.output_queue.get()
            if s16le_chunk is None:
                break

            f32le_chunk = convert_s16le_to_f32le(s16le_chunk)
            asyncio.run_coroutine_threadsafe(self.converted_audio_cb(f32le_chunk), loop)

        asyncio.run_coroutine_threadsafe(self.converted_audio_cb(None), loop)

    def _start_ffmpeg_process(self):
        return subprocess.Popen([
            'ffmpeg',
            # '-ar', f'{self.input_sample_rate}',
            '-i', 'pipe:0',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', f'{self.output_sample_rate}',
            '-ac', '1',
            'pipe:1'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def stop(self):
        if not self.running:
            return

        self.input_queue.put(None)

        if self.ffmpeg_process:
            self.ffmpeg_process.kill()
            self.ffmpeg_process.wait()

        for thread in self.threads:
            thread.join(timeout=5)

        self.running = False

    def is_running(self):
        return self.running


class BaseLLMAgent:
    def __init__(self, model_name, system_prompt, examples=None, output_json=False):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.examples = examples
        self.output_json = output_json

    def completion(self, context):
        assert isinstance(context, ConversationContext), f"Invalid context type {context.__class__}"
        
        messages = [
            {
              "role": "system",
              "content": self.system_prompt
            }
        ]

        if self.examples:
            for example in self.examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

        messages += context.get_messages(include_fields=["role", "content"])

        logger.error("Messages:\n{}", self._messages_to_text(messages))
        
        response = litellm.completion(
            model=self.model_name, 
            messages=messages, 
            response_format={"type": "json_object"} if self.output_json else None,
            temperature=0.5
        )

        r_content = response.choices[0].message.content
        logger.debug("Response content: {}", r_content)

        if self.output_json:
            response = json.loads(r_content)
        else:
            response = r_content

        return {
            "response": response,
            "messages": messages
        }

    @staticmethod
    def format_transcription(confirmed, unconfirmed):
        text = confirmed
        if unconfirmed:
            text += f" ({unconfirmed})"
        return text.strip()

    def _extra_context_to_text(self, context):
        # Override this in child class
        return ""

    def _messages_to_text(self, messages):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])


class ResponseLLMAgent(BaseLLMAgent):
    default_response = ""
    
    def __init__(self):
        super().__init__(
            model_name="openai/openai-gpt-4o-mini", 
            system_prompt=response_agent_system_prompt, 
            examples=response_agent_examples,
            output_json=False
        )

    # def _extra_context_to_text(self, context):
    #     assert "detection_agent_state" in context, "This agent requires detection agent's current state"
    #     assert "clarification_agent_state" in context, "This agent requires clarification agent's current state"
    #     d_state = context["detection_agent_state"]
    #     c_state = context["clarification_agent_state"]
    #     text = "Detection agent state:\nAction: {}\nReason: {}\n\n".format(d_state["action"], d_state["reason"])
    #     text += "Clarification agent state:\nHas question: {}\nQuestion: {}\n\n".format(c_state["has_question"], c_state["question"])
    #     return text

    def completion(self, context, *args, **kwargs):
        result = super().completion(context, *args, **kwargs)

        if result["response"] is None:
            result["response"] = self.default_response

        return result


class ConversationContext:
    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()

    def add_message(self, message, text_compare_f=None):
        with self.lock:
            assert message["role"].lower() in ["assistant", "user"], f"Unknown role {message['role']}"

            if not self.messages or self.messages[-1]["role"].lower() != message["role"].lower():
                message["id"] = len(self.messages)
                message["handled"] = False
                self.messages.append(message)
                return True

            if text_compare_f is None:
                text_compare_f = lambda x, y: x == y

            if not text_compare_f(self.messages[-1]["content"], message["content"]):
                self.messages[-1]["content"] = message["content"]
                self.messages[-1]["handled"] = False
                return True

            return False

    def get_messages(self, include_fields=None, filter=None):
        if include_fields is None:
            messages = self.messages
        else:
            messages = [{k: v for k, v in msg.items() if k in include_fields} for msg in self.messages]

        if filter:
            messages = [msg for msg in messages if filter(msg)]

        return messages

    def update_message(self, id, **kwargs):
        for msg in self.messages:
            if msg["id"] == id:
                msg.update(kwargs)
                return True
        return False

    def to_text(self):
        if not self.messages:
            return ""

        lines = []
        for item in self.messages:
            if not "time" in item:
                time_str = "00:00:00"
            else:
                time_str = item["time"].strftime("%H:%M:%S")
            
            role = item["role"].lower()
            content = item["content"]
            
            lines.append(f"{time_str} | {role} - {content}")
        
        return "\n".join(lines)

    def to_json(self, filter=None):
        messages = self.get_messages(filter=filter)
        return json.loads(json.dumps(messages, default=self._json_serializer))

    @staticmethod
    def _json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.strftime("%H:%M:%S")
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


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
        message = {
            "role": "assistant",
            "content": response_agent_greeting_message,
            "time": datetime.now()
        }
        self._update_conversation_context(message)

    @logger.catch
    async def handle_input_audio(self, audio_chunk):
        """ audio_chunk is f32le """
        end_of_audio = audio_chunk is None
        asr_result = self.asr.process_chunk(audio_chunk, finalize=end_of_audio)

        if asr_result and (asr_result["confirmed_text"] or asr_result["unconfirmed_text"]):
            message = self._create_message_from_transcription(asr_result)
            context_changed = self._update_conversation_context(message)
            # if context_changed:
            #     self._transcription_changed.set()

        if end_of_audio:
            need_response = self.conversation_context.messages and self.conversation_context.messages[-1]["role"] == "user"
            if need_response:
                response = self._response_agent.completion(self.conversation_context)
                self.voice_generator.generate_async(text=response["response"])
                agent_response = {
                    "role": "assistant",
                    "content": response["response"],
                    "time": datetime.now()
                }
                self._update_conversation_context(agent_response)

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
        context_changed = self.conversation_context.add_message(message, text_compare_f=compare_ignore_case)
        if context_changed:
            asyncio.run_coroutine_threadsafe(self.context_changed_cb(self.conversation_context), self._event_loop)
        return context_changed

    def _response_agent_loop(self):
        while True:
            self._transcription_changed.wait()
            self._transcription_changed.clear()

            if not self.running:
                break
            
            logger.error("Context changed:\n{}", self.conversation_context.to_text())
            response = self._response_agent.completion(self.conversation_context)
            logger.error("Agent response:\n{}", response["response"])

            self.last_agent_message = {
                "role": "assistant",
                "content": response["response"],
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

    @logger.catch
    async def handle_input_audio(audio_chunk):
        # audio_chunk is f32le
        await conversation.handle_input_audio(audio_chunk)
        if audio_chunk is not None:
            audio_input_saver.write(convert_f32le_to_s16le(audio_chunk))

    @logger.catch
    async def handle_context_changed(context):
        messages = context.to_json(filter=lambda msg: not msg["handled"])
        for msg in messages:
            logger.info("Sending message: {}", msg)
            await websocket.send(json.dumps(msg))
            context.update_message(msg["id"], handled=True)

    @logger.catch
    async def handle_generated_audio(audio_chunk):
        # audio_chunk is f32le
        if audio_chunk is None: # end of audio
            audio_output_stream.flush()
        else:
            audio_output_stream.put(audio_chunk)
            audio_output_saver.write(convert_f32le_to_s16le(audio_chunk))

    @logger.catch
    async def handle_webm_audio(audio_chunk):
        # chunk is webm
        if audio_chunk is not None and websocket.open:
            await websocket.send(audio_chunk)

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
        output_sample_rate=24000 # client expects 16000
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


async def main():
    logger.add(f"logs/server/server-{get_timestamp()}.log", rotation="100 MB")

    server = await websockets.serve(
        handle_connection,
        "0.0.0.0",
        8765,
        ssl=ssl_context
    )
    logger.info("WebSocket server started on wss://0.0.0.0:8765")

    await server.wait_closed()
    logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
