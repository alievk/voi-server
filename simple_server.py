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

import numpy as np
import torch
import websockets
from loguru import logger

from streaming import OnlineASR, SAMPLING_RATE


class WebmToPcmConverter:
    def __init__(self, audio_callback, chunk_size_ms=1000):
        self.ffmpeg_process = None
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.audio_callback = audio_callback
        self.chunk_size = int(chunk_size_ms / 1000 * SAMPLING_RATE * 2)
        self.webm_buffer = b''
        self.webm_file = None
        self.running = False
        self.threads = []

    def start(self):
        self.running = True
        self.ffmpeg_process = self._start_ffmpeg_process()
        self.threads = [
            threading.Thread(target=self._ffmpeg_in_pipe, name='ffmpeg-in-pipe', daemon=True),
            threading.Thread(target=self._ffmpeg_out_pipe, name='ffmpeg-out-pipe', daemon=True),
            threading.Thread(target=self._audio_callback, name='ffmpeg-callback', daemon=True, args=(asyncio.get_event_loop(),))
        ]
        for thread in self.threads:
            thread.start()
        self._create_webm_file()

    def put(self, chunk):
        self.input_queue.put(chunk)
        self.webm_buffer += chunk
        # write buffer to file every 1MB
        if len(self.webm_buffer) >= 1024 * 1024:
            self._flush_webm_buffer()

    def _ffmpeg_in_pipe(self):
        while self.running:
            chunk = self.input_queue.get()
            if chunk is None:
                break
            self.ffmpeg_process.stdin.write(chunk)
            self.ffmpeg_process.stdin.flush()

    def _ffmpeg_out_pipe(self):
        buffer = b''
        while self.running:
            pcm_chunk = self.ffmpeg_process.stdout.read(1024)
            if not pcm_chunk:
                break
            buffer += pcm_chunk
            if len(buffer) >= self.chunk_size:
                self.output_queue.put(buffer)
                buffer = b''

    def _audio_callback(self, loop):
        def convert_s16le_to_f32le(buffer):
            return np.frombuffer(buffer, dtype='<i2').flatten().astype(np.float32) / 32768.0

        while self.running:
            pcm_chunk = self.output_queue.get()
            if pcm_chunk is None:
                break
            pcm_chunk = convert_s16le_to_f32le(pcm_chunk)
            asyncio.run_coroutine_threadsafe(self.audio_callback(pcm_chunk), loop)

    def _start_ffmpeg_process(self):
        return subprocess.Popen([
            'ffmpeg',
            '-i', 'pipe:0',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', f'{SAMPLING_RATE}',
            '-ac', '1',
            'pipe:1'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _create_webm_file(self):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"audio_{timestamp}.webm"
        tmp_dir = 'logs/audio/webm'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.webm_file = open(os.path.join(tmp_dir, filename), 'wb')

    def _flush_webm_buffer(self):
        if self.webm_file:
            self.webm_file.write(self.webm_buffer)
            self.webm_file.flush()
        self.webm_buffer = b''

    def stop(self):
        self.running = False

        self.input_queue.put(None)
        self.output_queue.put(None)

        self._flush_webm_buffer()
        if self.webm_file:
            self.webm_file.close()

        if self.ffmpeg_process:
            self.ffmpeg_process.kill()
            self.ffmpeg_process.wait()

        for thread in self.threads:
            thread.join(timeout=5)


class Conversation:
    def __init__(self, handle_transcription_cb=None):
        self.handle_transcription_cb = handle_transcription_cb

        self.asr = OnlineASR()

    async def handle_audio(self, audio_chunk):
        result = self.asr.process_chunk(audio_chunk)

        # TODO: detect speech_id
        result["speech_id"] = 0

        if result and self.handle_transcription_cb:
            await self.handle_transcription_cb(result)


async def handle_connection(websocket):
    logger.info("New connection is started")

    async def handle_audio(audio_chunk):
        await conversation.handle_audio(audio_chunk)

    async def handle_transcription(data):
        try:
            data = json.dumps(data)
            await websocket.send(data)
        except Exception as e:
            logger.error(f"Error sending transcription: {e}")

    converter = WebmToPcmConverter(handle_audio)
    converter.start()

    conversation = Conversation(handle_transcription)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                converter.put(message)
            elif message == 'END_CONVERSATION':
                logger.info("Received END_CONVERSATION message")
                break
            else:
                logger.warning(f"Received unexpected message: {message}")
        logger.info("WebSocket connection closed, checking timeout")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed unexpectedly")
    finally:
        logger.info("Stopping audio converter")
        converter.stop()
    logger.info("Connection is done")


ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('localhost+2.pem', 'localhost+2-key.pem')


async def main():
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