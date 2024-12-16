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

from audio import adjust_speed

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from text import SentenceStream, split_text_into_speech_segments


class VoiceGeneratorBase:
    def generate(self, text, streaming=False):
        raise NotImplementedError

    def generate_async(self, text, id=None):
        raise NotImplementedError

    def maybe_set_voice_tone(self, voice_tone, role="character"):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    @property
    def sample_rate(self):
        raise NotImplementedError


class DummyVoiceGenerator(VoiceGeneratorBase):
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, text, streaming=False):
        zero_chunk = np.zeros(10000)
        if streaming:
            return itertools.repeat(zero_chunk)
        else:
            return zero_chunk

    def generate_async(self, text, id=None):
        pass

    def maybe_set_voice_tone(self, voice_tone, role="character"):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def sample_rate(self):
        return 24000


# OK voices for the multispeaker_original model:
# Barbora MacLean, Damjan Chapman, Luis Moray

class VoiceGenerator(VoiceGeneratorBase):
    _cached_tts_model_params = None

    def __init__(
        self,
        generated_audio_cb=None,
        model_name=None,
        voice=None,
        narrator_voice=None,
        mute_narrator=False,
        cached=False,
        voice_speed=1.0
    ):
        """ 
        generated_audio_cb receives f32le audio chunks 
        valid model_name values are listed in tts_models.json
        """
        self.generated_audio_cb = generated_audio_cb
        self.model_name = "multispeaker_original" if model_name is None else model_name
        self.language = 'en'
        self.tts_temperature = 0.7
        self.speed = voice_speed

        self.tts_model_params = self.get_model(self.model_name, cached=cached)
        self.tts_model = self.tts_model_params["model"]
        self.tts_voices = self.tts_model_params["voices"]

        self.mute_narrator = mute_narrator

        self.running = False
        self.text_queue = None
        self._event_loop = asyncio.get_event_loop()

        default_voice = list(self.tts_voices.keys())[0]
        self.set_voice(default_voice if voice is None else voice, role="character")
        self.set_voice(default_voice if narrator_voice is None else narrator_voice, role="narrator")

    def start(self):
        self.running = True
        self.text_queue = queue.Queue()
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.start()

    def set_model(self, model_name, cached=False):
        self.model_name = model_name
        self.tts_model, self.tts_voices = self.get_model(self.model_name, cached=cached)

    def set_voice(self, voice, role="character"):
        if voice not in self.tts_voices:
            raise ValueError(f"Voice {voice} not found. Available voices: {self.tts_voices.keys()}")

        assert role in ["character", "narrator"]
        if role == "character":
            self.voice = voice
        else:
            self.narrator_voice = voice

    def maybe_set_voice_tone(self, voice_tone, role="character"):
        if ("voice_tone_map" in self.tts_model_params and
            voice_tone in self.tts_model_params["voice_tone_map"]):
            self.set_voice(self.tts_model_params["voice_tone_map"][voice_tone], role=role)
        else:
            logger.warning(f"Voice tone {voice_tone} not found, ignoring")

    @staticmethod
    def get_model(model_name, cached=False):
        if (cached and 
            VoiceGenerator._cached_tts_model_params is not None and 
            VoiceGenerator._cached_tts_model_params["model_name"] == model_name):
            return VoiceGenerator._cached_tts_model_params

        if VoiceGenerator._cached_tts_model_params:
            VoiceGenerator._cached_tts_model_params["model"].cpu()
            del VoiceGenerator._cached_tts_model_params["model"]
            del VoiceGenerator._cached_tts_model_params["voices"]
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
            "avg_text_len": model_config["avg_text_len"],
            "speed": model_config.get("speed", 1.0)
        }
        if "voice_tone_map" in model_config:
            model_params["voice_tone_map"] = model_config["voice_tone_map"]

        VoiceGenerator._cached_tts_model_params = model_params

        return model_params

    def _sanitize_text(self, text):
        return text

    def generate(self, text, streaming=False):
        segments = split_text_into_speech_segments(text, avg_text_len=self.tts_model_params["avg_text_len"])

        text_chunks = []
        gpt_cond_latent = []
        speaker_embedding = []
        for segment in segments:
            if self.mute_narrator and segment["role"] == "narrator":
                continue
            text_chunks.append(segment["text"])
            voice = self.voice if segment["role"] == "character" else self.narrator_voice
            gpt_cond_latent.append(self.tts_voices[voice]["gpt_cond_latent"])
            speaker_embedding.append(self.tts_voices[voice]["speaker_embedding"])

        if streaming:
            return itertools.chain.from_iterable(
                self._stream_generator(
                    text_chunk,
                    gpt_cond_latent,
                    speaker_embedding
                )
                for text_chunk, gpt_cond_latent, speaker_embedding 
                in zip(text_chunks, gpt_cond_latent, speaker_embedding)
            )
        else:
            audio_chunks = []
            for text_chunk, gpt_cond_latent, speaker_embedding in zip(text_chunks, gpt_cond_latent, speaker_embedding):
                chunk = self.tts_model.inference(
                    text_chunk,
                    language=self.language,
                    temperature=self.tts_temperature,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    speed=self.speed
                )
                audio_chunks.append(chunk["wav"])

            return self._postprocess(np.concatenate(audio_chunks))

    def generate_async(self, text, id=None):
        if not self.running:
            raise RuntimeError("VoiceGenerator is not running")

        self.text_queue.put({"text": text, "id": id})

    def _processing_loop(self):
        while self.running:
            item = self.text_queue.get()
            if item is None:
                break

            text, id = item["text"], item["id"]

            if text.startswith("file:"):
                audio_chunk, sr = librosa.load(text[5:], sr=None)
                duration = len(audio_chunk) / sr
                asyncio.run_coroutine_threadsafe(self.generated_audio_cb(audio_chunk=audio_chunk, speech_id=id), self._event_loop)
            else:
                duration = 0.0
                for chunk in self.generate(text, streaming=True):
                    asyncio.run_coroutine_threadsafe(self.generated_audio_cb(audio_chunk=chunk, speech_id=id), self._event_loop)
                    duration += len(chunk) / self.sample_rate
                    if not self.running or not self.text_queue.empty(): # stop on new text or interrupt
                        break

            # end of generation
            asyncio.run_coroutine_threadsafe(self.generated_audio_cb(audio_chunk=None, speech_id=id, duration=duration), self._event_loop)

    def _stream_generator(
        self,
        text,
        gpt_cond_latent,
        speaker_embedding,
    ):
        audio_chunks = self.tts_model.inference_stream(
            text=text,
            language=self.language,
            temperature=self.tts_temperature,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=self.speed
        )

        for chunk in audio_chunks:
            yield self._postprocess(chunk.cpu().numpy())

    @property
    def sample_rate(self):
        return 24000

    def _postprocess(self, chunk):
        # I was not able to get speed control working for streaming audio,
        # so I'm just returning the chunk as is for now
        # return adjust_speed(chunk, self.tts_model_params["speed"], self.sample_rate)
        return chunk

    def stop(self):
        self.running = False
        self.text_queue.put(None)
        self.thread.join(timeout=5)
        self.text_queue = None
