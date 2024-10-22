import asyncio
import subprocess
import threading

import numpy as np

from loguru import logger


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
