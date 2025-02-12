import os
import asyncio
import queue
import threading
import json
import itertools
import librosa
from loguru import logger

import torch
import numpy as np

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from audio import FakeAudioStream
from text import split_text_into_speech_segments


cuda_lock = threading.Lock()


class VoiceGenerator:
    _model_cache = {}

    def __init__(
        self,
        model_name=None,
        voice=None,
        cached=False,
        voice_speed=None,
        leading_silence=None,
        trailing_silence=None
    ):
        """ 
        valid model_name values are listed in tts_models.json
        """
        self.model_name = "multispeaker_original" if model_name is None else model_name
        self.language = 'en'
        self.tts_temperature = 0.7
        self.speed = voice_speed if voice_speed is not None else 1.0
        self.leading_silence = 0.0 if leading_silence is None else leading_silence
        self.trailing_silence = 0.0 if trailing_silence is None else trailing_silence

        self.tts_model_params = self.get_model(self.model_name, cached=cached)
        self.tts_model = self.tts_model_params["model"]
        self.tts_voices = self.tts_model_params["voices"]

        default_voice = list(self.tts_voices.keys())[0]
        self.set_voice(default_voice if voice is None else voice)

    def start(self):
        self.running = True
        self.text_queue = queue.Queue()
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.start()

    def set_model(self, model_name, cached=False):
        self.model_name = model_name
        self.tts_model, self.tts_voices = self.get_model(self.model_name, cached=cached)

    def set_voice(self, voice):
        if voice not in self.tts_voices:
            raise ValueError(f"Voice {voice} not found. Available voices: {self.tts_voices.keys()}")

        self.voice = voice

    def maybe_set_voice_tone(self, voice_tone):
        if ("voice_tone_map" in self.tts_model_params and
            voice_tone in self.tts_model_params["voice_tone_map"]):
            self.set_voice(self.tts_model_params["voice_tone_map"][voice_tone])
        else:
            logger.warning(f"Voice tone {voice_tone} not found, ignoring")

    @staticmethod
    def get_model(model_name, cached=False):
        if model_name in VoiceGenerator._model_cache:
            if cached: # return cached model
                return VoiceGenerator._model_cache[model_name]
            else: # re-load the model
                VoiceGenerator._model_cache[model_name]["model"].cpu()
                del VoiceGenerator._model_cache[model_name]
                torch.cuda.empty_cache()

        model_file = os.path.join(os.path.dirname(__file__), "tts_models.json")
        with open(model_file, "r") as f:
            model_config = json.load(f)[model_name]

        # TTS lib will silently fail if the tokenizer file does not exist
        assert os.path.exists(model_config["tokenizer"]), f"Tokenizer file does not exist: {model_config['tokenizer']}"

        config = XttsConfig()
        config.load_json(model_config["config"])
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=model_config["checkpoint"],
            vocab_path=model_config["tokenizer"],
            use_deepspeed=False
        )
        model.cuda()

        voices = torch.load(model_config["voices"])

        model_params = {
            "model_name": model_name,
            "model": model,
            "voices": voices,
            "avg_text_len": model_config["avg_text_len"]
        }
        if "voice_tone_map" in model_config:
            model_params["voice_tone_map"] = model_config["voice_tone_map"]

        VoiceGenerator._model_cache[model_name] = model_params

        return model_params

    def _sanitize_text(self, text): 
        return text

    def _generate_silence(self, duration, streaming=False):
        if streaming:
            return itertools.repeat(np.zeros(int(duration * self.sample_rate), dtype=np.float32), 1)
        else:
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

    def generate(self, text, streaming=False):
        # TODO: split text into chunks

        if streaming:
            return itertools.chain.from_iterable([
                self._generate_silence(self.leading_silence, streaming=True),
                self._stream_generator(text),
                self._generate_silence(self.trailing_silence, streaming=True)
            ])
        else:
            with cuda_lock:
                audio = self.tts_model.inference(
                    text,
                    language=self.language,
                    temperature=self.tts_temperature,
                    gpt_cond_latent=self.tts_voices[self.voice]["gpt_cond_latent"],
                    speaker_embedding=self.tts_voices[self.voice]["speaker_embedding"],
                    speed=self.speed
                )["wav"]

            return np.concatenate([
                self._generate_silence(self.leading_silence),
                audio,
                self._generate_silence(self.trailing_silence)
            ])

    def _stream_generator(
        self,
        text
    ):
        audio_chunks = self.tts_model.inference_stream(
            text=text,
            language=self.language,
            temperature=self.tts_temperature,
            gpt_cond_latent=self.tts_voices[self.voice]["gpt_cond_latent"],
            speaker_embedding=self.tts_voices[self.voice]["speaker_embedding"],
            speed=self.speed
        )

        with cuda_lock:
            for chunk in audio_chunks:
                yield chunk.cpu().numpy()

    @property
    def sample_rate(self):
        return 24000

    def stop(self):
        self.running = False
        self.text_queue.put(None)
        self.thread.join(timeout=5)
        self.text_queue = None


class MultiVoiceGenerator:
    def __init__(
        self,
        generators
    ):
        assert isinstance(generators, dict), f"generators must be a \{'role': model\} like dict, but got {generators.__class__}"
        self.generators = generators

    @staticmethod
    def from_config(config, cached=False):
        generators = {}
        for role, params in config.items():
            generators[role] = VoiceGenerator(
                model_name=params["model"],
                voice=params.get("voice"),
                leading_silence=params.get("leading_silence"),
                trailing_silence=params.get("trailing_silence"),
                voice_speed=params.get("speed"),
                cached=cached,
            )

        return MultiVoiceGenerator(generators)

    def _get_segments(self, text):
        segments = split_text_into_speech_segments(text, avg_text_len=100)
        roles = set([s["role"] for s in segments])
        generators = set(self.generators.keys())
        if roles > generators:
            logger.warning(f"There aren't generators for these roles: {roles - generators}")
        return segments

    def generate(self, text, streaming=False):
        segments = self._get_segments(text)

        if streaming:
            return itertools.chain.from_iterable(
                self.generators[segment["role"]].generate(segment["text"], streaming=True)
                for segment in segments if segment["role"] in self.generators
            )
        else:
            audio_chunks = []
            for segment in segments:
                if segment["role"] in self.generators:
                    chunk = self.generators[segment["role"]].generate(segment["text"], streaming=False)
                    audio_chunks.append(chunk)

            return np.concatenate(audio_chunks)

    @property
    def sample_rate(self):
        return self.generators[list(self.generators.keys())[0]].sample_rate

    def maybe_set_voice_tone(self, voice_tone):
        assert "character" in self.generators, "Only character voice tone is supported for now, but character voice generator is not found"
        self.generators["character"].maybe_set_voice_tone(voice_tone)


class DummyVoiceGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, text, streaming=False):
        zero_chunk = np.zeros(int(1 * self.sample_rate)) # 1 second of silence
        if streaming:
            return itertools.repeat(zero_chunk, 1)
        else:
            return zero_chunk

    def maybe_set_voice_tone(self, voice_tone, role="character"):
        pass

    @property
    def sample_rate(self):
        return 24000


class AsyncVoiceGenerator:
    def __init__(
        self, 
        voice_generator, 
        generated_audio_cb, 
        error_cb=None
    ):
        """ 
        generated_audio_cb receives f32le audio chunks 
        """
        self.voice_generator = voice_generator
        self.generated_audio_cb = generated_audio_cb
        self.error_cb = error_cb
        self.running = False
        self.text_queue = None
        self._event_loop = asyncio.get_event_loop()

    def generate(self, text, id=None):
        if not self.running:
            raise RuntimeError("AsyncVoiceGenerator is not running")
        self.text_queue.put({"text": text, "id": id})

    def start(self):
        self.running = True
        self.text_queue = queue.Queue()
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        self.text_queue.put(None)
        self.thread.join(timeout=5)
        self.text_queue = None

    def _processing_loop(self):
        try:
            while self.running:
                item = self.text_queue.get()
                if item is None:
                    break

                text, id = item["text"], item["id"]

                if text.startswith("file:"):
                    stream = FakeAudioStream(text[5:], chunk_length=0.1, sr=self.sample_rate)
                    duration = stream.duration
                    for chunk, _ in stream:
                        asyncio.run_coroutine_threadsafe(
                            self.generated_audio_cb(audio_chunk=chunk, speech_id=id), 
                            self._event_loop
                        )
                else:
                    duration = 0.0
                    for chunk in self.voice_generator.generate(text, streaming=True):
                        asyncio.run_coroutine_threadsafe(
                            self.generated_audio_cb(audio_chunk=chunk, speech_id=id), 
                            self._event_loop
                        )
                        duration += len(chunk) / self.voice_generator.sample_rate
                        if not self.running or not self.text_queue.empty():
                            break

                asyncio.run_coroutine_threadsafe(
                    self.generated_audio_cb(audio_chunk=None, speech_id=id, duration=duration),
                    self._event_loop
                )
        except Exception as e:
            self.emit_error(e)
            raise e

    def emit_error(self, error):
        if self.error_cb:
            asyncio.run_coroutine_threadsafe(self.error_cb(error), self._event_loop)

    @property
    def sample_rate(self):
        return self.voice_generator.sample_rate

    def maybe_set_voice_tone(self, voice_tone):
        self.voice_generator.maybe_set_voice_tone(voice_tone)
