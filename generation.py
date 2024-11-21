import os
import asyncio
import queue
import threading
import json
import re
import librosa
from loguru import logger

import torch
import numpy as np

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from text import SentenceStream


# OK voices for the multispeaker_original model:
# Barbora MacLean, Damjan Chapman, Luis Moray

class VoiceGenerator:
    _cached_tts_model_params = None

    def __init__(self, generated_audio_cb=None, model_name=None, voice=None, cached=False):
        """ 
        generated_audio_cb receives f32le audio chunks 
        valid model_name values are listed in tts_models.json
        """
        self.generated_audio_cb = generated_audio_cb
        self.model_name = "multispeaker_original" if model_name is None else model_name
        self.language = 'en'
        self.tts_temperature = 0.7
        self.voice = voice

        self.tts_model_params = self.get_model(self.model_name, cached=cached)
        self.tts_model = self.tts_model_params["model"]
        self.tts_voices = self.tts_model_params["voices"]

        self.running = False
        self.text_queue = None
        self._event_loop = asyncio.get_event_loop()

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
            raise ValueError(f"Voice {voice} not found")

        self.voice = voice

    def maybe_set_voice_tone(self, voice_tone):
        if ("voice_tone_map" in self.tts_model_params and
            voice_tone in self.tts_model_params["voice_tone_map"]):
            self.set_voice(self.tts_model_params["voice_tone_map"][voice_tone])
        else:
            logger.warning(f"Voice tone {voice_tone} not found, ignoring")

    @staticmethod
    def get_model(model_name, cached=False):
        if (cached and 
            VoiceGenerator._cached_tts_model_params is not None and 
            VoiceGenerator._cached_tts_model_params["model_name"] == model_name):
            return VoiceGenerator._cached_tts_model_params

        model_file = os.path.join(os.path.dirname(__file__), "tts_models.json")
        with open(model_file, "r") as f:
            model_config = json.load(f)[model_name]

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
            "max_context_len": model_config["max_context_len"]
        }
        if "voice_tone_map" in model_config:
            model_params["voice_tone_map"] = model_config["voice_tone_map"]

        VoiceGenerator._cached_tts_model_params = model_params

        return model_params

    def _sanitize_text(self, text):
        # remove *whispers* and (whispers)
        pattern = r"\*.*?\*|\(.*?\)"
        return re.sub(pattern, "", text)

    def _split_text(self, text):
        """
        Split text into chunks that fit the TTS model context length.
        """
        max_context_len = self.tts_model_params["max_context_len"]
        text_chunks = []
        buffer = ""

        for sent in SentenceStream(text):
            if not buffer:
                buffer = sent
                continue
                
            if len(buffer) + len(sent) <= max_context_len:
                buffer += sent
            else:
                text_chunks.append(buffer)
                buffer = sent

        if buffer:
            text_chunks.append(buffer)

        return text_chunks

    def generate(self, text, voice=None, streaming=False):
        if voice:
            self.set_voice(voice)

        if self.voice is None:
            voice = list(self.tts_voices.keys())[0]
            self.set_voice(voice)
            logger.warning(f"No voice specified, using the first available voice: {voice}")

        gpt_cond_latent = self.tts_voices[self.voice]["gpt_cond_latent"]
        speaker_embedding = self.tts_voices[self.voice]["speaker_embedding"]

        text = self._sanitize_text(text)

        if streaming:
            return self._stream_generator(
                text=text,
                language=self.language,
                temperature=self.tts_temperature,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
            )
        else:
            text_chunks = self._split_text(text)
            audio_chunks = []
            for text_chunk in text_chunks:
                chunk = self.tts_model.inference(
                    text_chunk,
                    language=self.language,
                    temperature=self.tts_temperature,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                )
                audio_chunks.append(chunk["wav"])

            return np.concatenate(audio_chunks)

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
        **kwargs
    ):
        text = kwargs.pop("text")
        text_chunks = self._split_text(text)
        for text_chunk in text_chunks:
            audio_chunks = self.tts_model.inference_stream(text=text_chunk, **kwargs)

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
