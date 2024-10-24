import os
import asyncio
import subprocess
import threading
import multiprocessing
import wave

import numpy as np

from loguru import logger


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
                self.ffmpeg_process.stdin.close()
                break
            self.ffmpeg_process.stdin.write(chunk)
            self.ffmpeg_process.stdin.flush()

    def _ffmpeg_out_pipe(self):
        buffer = bytearray()
        while self.running:
            pcm_chunk = self.ffmpeg_process.stdout.read(1024)
            if pcm_chunk == b'':
                logger.debug("ffmpeg stdout: EOF")
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
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self):
        if not self.running:
            return

        self.input_queue.put(None)

        if self.ffmpeg_process:
            self.ffmpeg_process.wait()

        for thread in self.threads:
            thread.join(timeout=5)

        self.running = False

    def is_running(self):
        return self.running


class AudioOutputStream:
    def __init__(self, converted_audio_cb, input_sample_rate=16000, output_sample_rate=16000):
        """ converted_audio_cb receives webm audio chunks """
        self.converted_audio_cb = converted_audio_cb
        self._event_loop = asyncio.get_event_loop()

        self.ffmpeg_process = None
        self.out_audio_thread = None
        self.running = False

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

        self.output_chunk_size = 1024

    def start(self):
        self.running = True
        self.ffmpeg_process = self._start_ffmpeg_process()
        self.out_audio_thread = threading.Thread(target=self._ffmpeg_out_pipe, name='ffmpeg-out-pipe', daemon=True)
        self.out_audio_thread.start()

    def put(self, chunk):
        """ chunk is f32le """
        if not self.running:
            logger.warning("AudioOutputStream is not running")
            return

        def float32_to_int16(float32_array):
            return np.int16(float32_array * 32768.0)

        # TODO: this is a blocking call, we should use a queue
        s16le_chunk = float32_to_int16(chunk).tobytes()
        self.ffmpeg_process.stdin.write(s16le_chunk)
        self.ffmpeg_process.stdin.flush()

    def _start_ffmpeg_process(self):
        return subprocess.Popen([
            "ffmpeg",
            "-f", "s16le",
            "-ar", str(self.input_sample_rate),
            "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-ar", str(self.output_sample_rate),
            "-f", "webm",
            "pipe:1"
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _ffmpeg_out_pipe(self):
        while self.running:
            chunk = self.ffmpeg_process.stdout.read(512)
            if chunk == b'':
                logger.debug("ffmpeg stdout: EOF")
                break

            asyncio.run_coroutine_threadsafe(self.converted_audio_cb(chunk), self._event_loop)

        asyncio.run_coroutine_threadsafe(self.converted_audio_cb(None), self._event_loop)

    def stop(self):
        if not self.running:
            return

        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()
        self.out_audio_thread.join(timeout=5)
        self.running = False

    def flush(self):
        self.stop()
        self.start()
