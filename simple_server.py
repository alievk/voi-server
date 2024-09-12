import asyncio
import websockets
import ssl
import logging
import ffmpeg
import os
from datetime import datetime
import numpy as np
import wave
from collections import deque
import webrtcvad
import subprocess
import multiprocessing
import threading

import sys, os
sys.path.append(os.path.dirname(__file__))
import voicebot_audio_api.src.main as vba

vba.CACHE_ROOT = '/home/user/.cache'
model = vba.get_whisper_model()

logging.getLogger().setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

def save_audio_to_file(chunks):
    filename = f"audio_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"
    audio_data = b''.join(chunks)

    tmp_dir = 'logs/audio/chunks'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    filename = os.path.join(tmp_dir, filename)
    
    # Save PCM 16-bit audio
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for 16-bit
        wf.setframerate(48000)
        wf.writeframes(b''.join(chunks))

    return filename
        
    # try:
    #     process = (
    #         ffmpeg
    #         .input('pipe:0', format='webm')
    #         .output('recorded_audio.wav', format='wav', acodec='pcm_s16le', ar=48000, ac=1)
    #         .overwrite_output()
    #         .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    #     )
        
    #     stdout, stderr = process.communicate(input=audio_data)
        
    #     if process.returncode == 0:
    #         logging.info("Audio saved to recorded_audio.wav")
    #     else:
    #         logging.error(f"FFmpeg error: {stderr.decode()}")
    # except ffmpeg.Error as e:
    #     logging.error(f"FFmpeg error: {str(e)}")
    # except Exception as e:
    #     logging.error(f"Error saving audio: {str(e)}")

    # return "recorded_audio.wav"

class WebmToPcmConverter:
    def __init__(self, callback, loop):
        self.process = None
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.callback = callback
        self.loop = loop
        self.webm_buffer = b''
        self.webm_file = None

    def start(self):
        self.process = multiprocessing.Process(target=self._run)
        self.process.start()
        threading.Thread(target=self._process_output, daemon=True).start()
        self._create_webm_file()

    def _create_webm_file(self):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"audio_{timestamp}.webm"
        tmp_dir = 'logs/audio/webm'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.webm_file = open(os.path.join(tmp_dir, filename), 'wb')

    def put(self, chunk):
        self.input_queue.put(chunk)
        self.webm_buffer += chunk
        if len(self.webm_buffer) >= 1024 * 1024:  # 1MB buffer
            self._write_webm_buffer()

    def _write_webm_buffer(self):
        if self.webm_file:
            self.webm_file.write(self.webm_buffer)
            self.webm_file.flush()
        self.webm_buffer = b''

    def _run(self):
        logging.info("WebmToPcmConverter _run")
        ffmpeg_process = self._start_ffmpeg_process()
        buffer = b''
        chunk_size = 48000 * 2  # 1 second of audio at 48kHz, 16-bit

        def send_chunks():
            while True:
                chunk = self.input_queue.get()
                logging.debug(f"WebmToPcmConverter Received chunk: {len(chunk)} bytes")
                ffmpeg_process.stdin.write(chunk)
                ffmpeg_process.stdin.flush()

        def receive_chunks():
            nonlocal buffer
            while True:
                pcm_chunk = ffmpeg_process.stdout.read(1024)
                if not pcm_chunk:
                    break
                buffer += pcm_chunk
                if len(buffer) >= chunk_size:
                    logging.info(f"WebmToPcmConverter Sending {len(buffer)} bytes to output queue")
                    self.output_queue.put(buffer)
                    buffer = b''

        send_thread = threading.Thread(target=send_chunks)
        receive_thread = threading.Thread(target=receive_chunks)

        send_thread.start()
        receive_thread.start()

        send_thread.join()
        receive_thread.join()

    def _process_output(self):
        while True:
            pcm_chunk = self.output_queue.get()
            logging.info(f"Processing output chunk of {len(pcm_chunk)} bytes")
            asyncio.run_coroutine_threadsafe(self.callback(pcm_chunk), self.loop)

    def _start_ffmpeg_process(self):
        return subprocess.Popen([
            'ffmpeg',
            '-i', 'pipe:0',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', '48000',
            '-ac', '1',
            'pipe:1'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self):
        self._write_webm_buffer()  # Write any remaining data
        if self.webm_file:
            self.webm_file.close()
        self.process.terminate()

class Transcriber:
    def __init__(self):
        self.model = vba.get_whisper_model()

    async def transcribe(self, segment):
        filename = save_audio_to_file([segment])
        iterator = vba.async_transcribe(path=filename, model=self.model, language='ru')
        skip_first = False
        try:
            async for text_chunk in iterator:
                if not skip_first:
                    skip_first = True
                    continue
                logging.debug(f"Transcribed text: {text_chunk['text']}")
                return text_chunk['text']
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}", exc_info=True)
            return None

class Conversation:
    def __init__(self, websocket, timeout=60):
        self.websocket = websocket
        self.timeout = timeout
        self.last_audio_time = None
        self.vad = webrtcvad.Vad(3)  # Aggressiveness is 3
        self.transcriber = Transcriber()
        self.current_utterance = []
        self.current_utterance_id = 0
        self.pcm_buffer = b''

    async def handle_audio(self, pcm_chunk):
        logging.info(f"Conversation handle_audio called with {len(pcm_chunk)} bytes")
        self.last_audio_time = asyncio.get_event_loop().time()
        self.pcm_buffer += pcm_chunk
        
        segment_size = 48000 * 10  # 1 second at 48kHz, 16-bit audio
        
        logging.debug(f"PCM buffer size: {len(self.pcm_buffer)} bytes")
        
        if len(self.pcm_buffer) < segment_size:
            return
        
        segment = self.pcm_buffer[:segment_size]
        self.pcm_buffer = self.pcm_buffer[segment_size:]

        # is_speech = self.vad.is_speech(segment, 48000)
        # logging.debug(f"VAD result: {'speech' if is_speech else 'non-speech'}")
        is_speech = True
        
        if is_speech:
            text = await self.transcriber.transcribe(segment)
            if text:
                logging.info(f"Transcribed text: {text}")
                self.current_utterance.append(text)
                await self.send_partial_utterance()
        else:
            logging.debug("Finalizing utterance due to non-speech")
            await self.finalize_utterance()

    async def send_partial_utterance(self):
        combined_text = " ".join(self.current_utterance)
        await self.websocket.send(f"{self.current_utterance_id}:{combined_text}")

    async def finalize_utterance(self):
        if self.current_utterance:
            combined_text = " ".join(self.current_utterance)
            await self.websocket.send(f"FINAL:{self.current_utterance_id}:{combined_text}")
            self.current_utterance = []
            self.current_utterance_id += 1

    async def check_timeout(self):
        if self.last_audio_time and asyncio.get_event_loop().time() - self.last_audio_time > self.timeout:
            await self.end_conversation()

    async def end_conversation(self):
        await self.websocket.close()

async def handle_connection(websocket):
    conversation = Conversation(websocket)

    async def audio_callback(pcm_chunk):
        logging.info(f"Audio callback called with {len(pcm_chunk)} bytes")
        try:
            await conversation.handle_audio(pcm_chunk)
        except Exception as e:
            logging.error(f"Error in handle_audio: {e}", exc_info=True)

    loop = asyncio.get_event_loop()
    converter = WebmToPcmConverter(audio_callback, loop)
    converter.start()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                logging.debug(f"Received audio chunk: {len(message)} bytes")
                converter.put(message)
            elif message == 'END_CONVERSATION':
                logging.info("Received END_CONVERSATION message")
                await conversation.end_conversation()
                break
            else:
                logging.warning(f"Received unexpected message: {message}")
        logging.info("WebSocket connection closed, checking timeout")
        await conversation.check_timeout()
    except websockets.exceptions.ConnectionClosed:
        logging.info("WebSocket connection closed unexpectedly")
        await conversation.end_conversation()
    finally:
        logging.info("Stopping audio converter")
        converter.stop()
    logging.info("Done")

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('localhost+2.pem', 'localhost+2-key.pem')

async def main():
    server = await websockets.serve(
        handle_connection,
        "0.0.0.0",
        8765,
        ssl=ssl_context
    )
    logging.info("WebSocket server started on wss://0.0.0.0:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())