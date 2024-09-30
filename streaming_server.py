import argparse
import socket
import sys
from loguru import logger
import soundfile
import librosa
import numpy as np
import io
import os
from datetime import datetime
import wave
import signal

from streaming import OnlineASR, SAMPLING_RATE


class AudioSaver:
    def __init__(self, log_dir="./logs/wav"):
        self.log_dir = log_dir
        self.wav_file = None
        self._setup_wav_file()

    def _setup_wav_file(self):
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_stream_{timestamp}.wav"
        filepath = os.path.join(self.log_dir, filename)
        
        logger.info(f"Logging audio to {filepath}")
        self.wav_file = wave.open(filepath, 'wb')
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(2)  # 16-bit audio
        self.wav_file.setframerate(SAMPLING_RATE)

    def write(self, raw_bytes):
        self.wav_file.writeframes(raw_bytes)

    def close(self):
        if self.wav_file:
            self.wav_file.close()


class AudioBuffer:
    def __init__(self, conn, save_audio=True):
        self.conn = conn
        self.buffer = b""
        self.save_audio = save_audio
        self.audio_saver = None

        if self.save_audio:
            self.audio_saver = AudioSaver()

    def read(self, chunk_size):
        while len(self.buffer) < chunk_size:
            bytes = self.conn.recv(4096)
            if not bytes:
                return None # end of stream
            self.buffer += bytes

        chunk = self.buffer[:chunk_size]
        self.buffer = self.buffer[chunk_size:]

        if self.save_audio:
            self.audio_saver.write(chunk)

        return self._bytes_to_audio(chunk)

    def _bytes_to_audio(self, raw_bytes):
        return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def close(self):
        if self.audio_saver:
            self.audio_saver.close()


class StreamingServer:
    def __init__(self, host, port, asr, chunk_size_ms=1000):
        self.host = host
        self.port = port
        self.asr = asr
        self.chunk_size = int(chunk_size_ms / 1000 * SAMPLING_RATE * 2) # 2 bytes per sample

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))

        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        logger.info("Ctrl+C received. Cleaning up...")
        self.stop()
        sys.exit(0)

    @logger.catch
    def run(self):
        self.sock.listen(1)
        logger.info(f"Listening on {self.host}:{self.port}")
        while self.running:
            conn, addr = self.sock.accept()
            logger.info(f"New connection from {addr}")
            self.handle_connection(conn)

    @logger.catch
    def handle_connection(self, conn):
        self.asr.reset()
        audio_buffer = AudioBuffer(conn)
        try:
            while True:
                audio = audio_buffer.read(self.chunk_size) # blocks until chunk_size bytes are read
                if audio is None:
                    break # end of stream
                result = self.asr.process_chunk(audio)
                if result:
                    self.process_asr_result(result)
        except ConnectionResetError:
            pass
        finally:
            conn.close()
            audio_buffer.close()
            logger.info("Connection closed")

    def process_asr_result(self, result):
        # Process the ASR result here
        logger.info(f"ASR Result: {result}")

    def stop(self):
        self.running = False
        self.sock.close()
        logger.info("Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=43007, help="Port to listen on")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="DEBUG", format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    asr = OnlineASR()
    server = StreamingServer(args.host, args.port, asr)
    server.run()
