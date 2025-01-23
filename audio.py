import os
import asyncio
import subprocess
import threading
import multiprocessing
import wave
from audiostretchy.stretch import AudioStretch
import soundfile as sf
import librosa
import io
import torchaudio
import torch

import numpy as np

from loguru import logger


def load_audio(fname, sr=None, duration=None):
    return librosa.load(fname, sr=sr, dtype=np.float32, duration=duration)


def convert_s16le_to_f32le(buffer):
    return np.frombuffer(buffer, dtype='<i2').flatten().astype(np.float32) / 32768.0


def convert_s16le_to_ogg(buffer, sample_rate):
    # opus codec always encodes in 48kHz
    # data = convert_s16le_to_f32le(buffer)
    # data = torchaudio.functional.resample(torch.from_numpy(data).unsqueeze(0), sample_rate, 48000)
    data = torch.frombuffer(buffer, dtype=torch.int16).unsqueeze(0)
    file = io.BytesIO()
    torchaudio.save(file, data, sample_rate, format="opus")
    return file.getvalue()


def convert_f32le_to_s16le(buffer):
    return (buffer * 32768.0).astype(np.int16).tobytes()


class WavSaver:
    def __init__(self, filename, channels=1, sample_width=2, sample_rate=16000, buffer_size=None):
        self.filename = filename
        self.channels = channels
        self.sample_width = sample_width
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.wav_file = None
        self.buffer = b''
        self._create_wav_file()

    def write(self, chunk):
        """ Input format is s16le (buffer) or f32le (numpy array) """
        if isinstance(chunk, np.ndarray) and np.issubdtype(chunk.dtype, np.floating):
            chunk = convert_f32le_to_s16le(chunk)

        self.buffer += chunk
        if self.buffer_size and len(self.buffer) >= self.buffer_size:
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
        self.wav_file = None


class WavGroupSaver:
    def __init__(self, dirname, **wav_kwargs):
        self.dirname = dirname
        self.wav_files = {}
        self.wav_kwargs = wav_kwargs
        os.makedirs(dirname, exist_ok=True)

    def write(self, chunk, filename):
        if filename not in self.wav_files:
            self.wav_files[filename] = WavSaver(os.path.join(self.dirname, filename), **self.wav_kwargs)
        self.wav_files[filename].write(chunk)

    def get_buffer(self, filename):
        return self.wav_files[filename].buffer

    def close(self, filename=None):
        if filename is None:
            for wav_file in self.wav_files.values():
                wav_file.close()
        elif filename in self.wav_files:
            self.wav_files[filename].close()
            del self.wav_files[filename]


class FakeAudioStream:
    def __init__(self, filename, chunk_length=0.1, duration=None, sr=None):
        """
        - chunk_length: length of the chunk to read in seconds
        - duration: maximum duration of the audio to read in seconds
        """
        self.audio, self.sr = load_audio(filename, sr=sr, duration=duration)
        self.duration = self.audio.shape[0] / self.sr
        self.chunk_length = chunk_length
        self.beg = 0
        logger.debug("Loaded audio is {:.2f} seconds", self.duration)

    def __iter__(self):
        return self

    def __next__(self):
        chunk, beg = self.read(self.chunk_length)
        if chunk is None:
            raise StopIteration
        return chunk, beg

    def read(self, chunk_length):
        beg = self.beg
        end = min(self.duration, beg + chunk_length)
        chunk = None
        if beg < end:
            beg_idx = int(beg * self.sr)
            end_idx = int(end * self.sr)
            chunk = self.audio[beg_idx:end_idx]
            self.beg = end
        else:
            logger.debug("End of stream")
        return chunk, beg

    def fast_forward(self, time):
        self.beg = min(self.duration, time)


class AudioInputStream:
    def __init__(
        self, 
        converted_audio_cb,
        output_chunk_size_ms=1000, 
        input_sample_rate=16000, 
        output_sample_rate=16000,
        input_format="webm"
    ):
        """
        Converts webm/mp4/ogg/pcm16 to f32le.
        """
        assert input_format in ["webm", "mp4", "ogg", "pcm16"]
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.ffmpeg_process = None
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.converted_audio_cb = converted_audio_cb
        self.output_chunk_size = int(output_chunk_size_ms / 1000 * self.output_sample_rate * 2)
        self.input_format = input_format
        self.running = False
        self.threads = []

    def start(self):
        if self.running:
            return

        self.threads = [
            threading.Thread(target=self._buffering, name='buffering', daemon=True),
            threading.Thread(target=self._audio_callback, name='audio-callback', daemon=True, args=(asyncio.get_event_loop(),))
        ]

        if self.input_format != "pcm16":
            self.ffmpeg_process = self._start_ffmpeg_process()
            self.threads.insert(0, threading.Thread(target=self._ffmpeg_in_pipe, name='ffmpeg-in-pipe', daemon=True))

        self.running = True # SET BEFORE STARTING THREADS

        for thread in self.threads:
            thread.start()

    def put(self, chunk):
        """ chunk is webm/mp4/ogg/pcm16 """
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

    def _buffering(self):
        buffer = bytearray()
        while self.running:
            if self.input_format == "pcm16":
                pcm_chunk = self.input_queue.get()
            else:
                pcm_chunk = self.ffmpeg_process.stdout.read(1024)
            if not pcm_chunk:
                logger.debug("Input audio EOF")
                break
            
            buffer.extend(pcm_chunk)
            while len(buffer) >= self.output_chunk_size:
                self.output_queue.put(bytes(buffer[:self.output_chunk_size]))
                del buffer[:self.output_chunk_size]
        
        if buffer:
            self.output_queue.put(bytes(buffer))

        self.output_queue.put(None)

    def _audio_callback(self, loop):
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
    def __init__(
        self, 
        converted_audio_cb, 
        input_sample_rate=16000, 
        output_sample_rate=16000,
        output_format="webm"
    ):
        """
        Asyncronous f32le to webm/mp4/ogg conversion.
        """
        assert output_format in ["webm", "mp4", "ogg"]
        self.converted_audio_cb = converted_audio_cb
        self._event_loop = asyncio.get_event_loop()

        self.ffmpeg_process = None
        self.out_audio_thread = None
        self.running = False

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_format = output_format

    def start(self):
        if self.running:
            return

        self.running = True # SET BEFORE STARTING THREADS

        self.ffmpeg_process = self._start_ffmpeg_process()
        self.out_audio_thread = threading.Thread(target=self._audio_callback, name='audio-callback', daemon=True)
        self.out_audio_thread.start()

    def put(self, chunk, speech_id=None):
        """ chunk is f32le """
        if not self.running:
            logger.warning("AudioOutputStream is not running")
            return

        # TODO: these are blocking calls, we should use a queue
        s16le_chunk = convert_f32le_to_s16le(chunk)
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

    def _audio_callback(self):
        while self.running:
            chunk = self.ffmpeg_process.stdout.read(512)

            if not chunk:
                logger.debug("Output audio EOF")
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


def adjust_speed(audio, speed, sample_rate):
    input_file = io.BytesIO()
    sf.write(input_file, audio, sample_rate, format="wav")
    input_file.seek(0)

    audio_stretch = AudioStretch()
    audio_stretch.open(file=input_file, format="wav")
    
    audio_stretch.stretch(
        ratio=1.0 / speed,
        gap_ratio=1.2,
        upper_freq=333,
        lower_freq=55,
        buffer_ms=25,
        threshold_gap_db=-40,
        double_range=False,
        fast_detection=False,
        normal_detection=False,
    )

    output_buffer = io.BytesIO()
    output_buffer.close = lambda: None
    audio_stretch.save(file=output_buffer, format="wav")

    output_buffer.seek(0)
    processed_audio, _ = sf.read(output_buffer, dtype="float32")

    return processed_audio