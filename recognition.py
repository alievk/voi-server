import numpy as np
import threading
from loguru import logger

import torch
import whisperx


SAMPLING_RATE = 16000 # Hz

MIN_AUDIO_BUFFER_DURATION = 1 # seconds
MAX_AUDIO_BUFFER_DURATION = 15 # seconds    

FAST_FORWARD_TIME_MARGIN = 0.1 # seconds

ASR_CONTEXT_LENGTH = 200 # words


cuda_lock = threading.Lock()


class AudioBuffer:
    def __init__(
        self, 
        sampling_rate=SAMPLING_RATE, 
        min_duration=MIN_AUDIO_BUFFER_DURATION, 
        max_duration=MAX_AUDIO_BUFFER_DURATION
    ):
        self.sampling_rate = sampling_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_size = int(min_duration * sampling_rate)
        self.max_size = int(max_duration * sampling_rate)
        self.buffer = None
        self.offset = None # global time of the buffer's first element
        self.clear()

    def push(self, data):
        assert isinstance(data, np.ndarray), f"Invalid data type: {type(data)}"
        self.buffer = np.concatenate([self.buffer, data])
        if len(self.buffer) < self.min_size:
            logger.debug("Audio buffer ({:.2f}s) is shorter than {}s", len(self.buffer) / self.sampling_rate, self.min_duration)
            return None, None
        elif len(self.buffer) > self.max_size:
            o_offset = self.offset
            logger.debug("Audio buffer ({:.2f}s) is longer than {}s, trimming.", len(self.buffer) / self.sampling_rate, self.max_duration)
            self.offset += (len(self.buffer) - self.max_size) / self.sampling_rate
            self.buffer = self.buffer[-self.max_size:]
            logger.debug("New buffer length: {:.2f}s. Offset: {:.2f}s -> {:.2f}s", len(self.buffer) / self.sampling_rate, o_offset, self.offset)
        return self.buffer, self.offset

    def clear(self):
        buffer, offset = self.buffer, self.offset
        self.buffer = np.empty((0,), dtype=np.float32)
        self.offset = 0
        return buffer, offset

    def fast_forward(self, time):
        if time < self.offset:
            logger.debug("Ignoring negative fast forward: {:.2f} -> {:.2f}s. Maybe the buffer was front-trimmed?", self.offset, time)
            return
        ff = min(time - self.offset, len(self.buffer) / self.sampling_rate)
        ff_size = int(ff * self.sampling_rate)
        self.buffer = self.buffer[ff_size:]
        self.offset += ff

    def empty(self):
        return len(self.buffer) == 0

class Word:
    def __init__(self, word, start, end, sep=""):
        self.word = word
        self.start = start
        self.end = end
        self.sep = sep

    @staticmethod
    def none():
        return Word(word=None, start=None, end=None)

    @staticmethod
    def to_text(words):
        words = words if isinstance(words, list) else [words]
        if not words:
            return ""
        return words[0].sep.join([w.word for w in words])

    @staticmethod
    def apply_offset(words, offset):
        words = words if isinstance(words, list) else [words]
        for i in range(len(words)):
            words[i].start += offset
            words[i].end += offset
        return words
        
    def __repr__(self):
        return f"{self.start}:{self.end} {self.word}"


class HypothesisBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.confirmed_words = []
        self.unconfirmed_words = []

    def update(self, current_words):
        if self.confirmed_words:
            current_words = self._fast_forward(current_words, self.confirmed_words[-1].end)
            self.unconfirmed_words = self._fast_forward(self.unconfirmed_words, self.confirmed_words[-1].end)
        if self.unconfirmed_words:
            lcp = self._longest_common_prefix(self.unconfirmed_words, current_words)
            self.confirmed_words.extend(lcp)

            if lcp:
                current_words = self._fast_forward(current_words, self.confirmed_words[-1].end)

        self.unconfirmed_words = current_words

        return self.confirmed_words, self.unconfirmed_words

    @staticmethod
    def _longest_common_prefix(s1, s2):
        min_len = min(len(s1), len(s2))
        logger.debug("comparing:")
        logger.debug("unconf: {}", Word.to_text(s1))
        logger.debug("currnt: {}", Word.to_text(s2))

        def compare(w1, w2):
            w1_lower = ''.join(c for c in w1.word.lower() if c.isalpha())
            w2_lower = ''.join(c for c in w2.word.lower() if c.isalpha())
            return w1_lower == w2_lower
        
        index = 0
        while index < min_len and compare(s1[index], s2[index]):
            index += 1

        return s1[:index]

    @staticmethod
    def _fast_forward(words, time):
        # fast forward `words` till the first word which starts after `time`
        while words and words[0].start < time:
            words.pop(0)
        return words


class VoiceActivityDetector:
    def __init__(self, sampling_rate=16000, voice_threshold=0.5):
        assert sampling_rate in [8000, 16000], f"Unsupported sampling rate {sampling_rate}"
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.sampling_rate = sampling_rate
        self.voice_threshold = voice_threshold
        self.silence_threshold = max(self.voice_threshold - 0.15, 0.1)

        self._sample_width = {8000: 256, 16000: 512}[self.sampling_rate]
        self._trailing_silence = 0.0
        self._trailing_voice = 0.0
        self._has_voice = None
        self._remainder = None
        self.reset()

    def reset(self):
        self._trailing_silence = 0.0
        self._trailing_voice = 0.0
        self._has_voice = False
        self._remainder = np.empty((0,), dtype=np.float32)

    def process_chunk(self, chunk):
        chunk = np.concatenate([self._remainder, chunk])

        for i in range(len(chunk) // self._sample_width):
            sample = chunk[i * self._sample_width:(i + 1) * self._sample_width]
            self._process_sample(sample)

        self._remainder = chunk[-(len(chunk) % self._sample_width):]

        return {
            "trailing_silence": self._trailing_silence,
            "trailing_voice": self._trailing_voice,
            "has_voice": self._has_voice
        }

    def _process_sample(self, chunk):
        prob = self.model(torch.from_numpy(chunk), self.sampling_rate)

        if prob > self.voice_threshold:
            self._trailing_voice += self._get_duration(chunk)
            self._trailing_silence = 0.0
            self._has_voice = True
        elif prob < self.silence_threshold:
            self._trailing_voice = 0.0
            self._trailing_silence += self._get_duration(chunk)

    def _get_duration(self, chunk):
        return len(chunk) / self.sampling_rate


class OfflineASR:
    word_sep = " "
    _cached_model_params = None

    @staticmethod
    def get_model(language="en", device="cuda", cached=True):
        if (cached and OfflineASR._cached_model_params and 
            OfflineASR._cached_model_params["language"] == language):
            return OfflineASR._cached_model_params
        
        if OfflineASR._cached_model_params:
            OfflineASR._cached_model_params["model"].model.model.unload_model(to_cpu=True)
            OfflineASR._cached_model_params["model_a"].cpu()
            del OfflineASR._cached_model_params["model"]
            del OfflineASR._cached_model_params["model_a"]
            torch.cuda.empty_cache()

        whisper_model = whisperx.load_model(
            "large-v2",
            device=device, 
            compute_type="float16", 
            language=language,
            asr_options={"suppress_numerals": True}
            # vad_options={'vad_onset': 0.8, 'vad_offset': 0.8}
        )

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language, 
            device=device, 
            model_name="WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "en" else None
        )

        OfflineASR._cached_model_params = {
            "model": whisper_model,
            "model_a": align_model,
            "align_metadata": align_metadata,
            "language": language
        }
        return OfflineASR._cached_model_params
    
    def __init__(self, language="en", cached=False):
        self.language = language
        self.batch_size = 1 # reduce if low on GPU mem
        self.device = "cuda"

        model_params = OfflineASR.get_model(language=language, cached=cached)
        self.model = model_params["model"]
        self.model_a = model_params["model_a"]
        self.align_metadata = model_params["align_metadata"]

    def transcribe(self, audio, previous_text=None):
        with cuda_lock:
            transcript = self.model.transcribe(
                audio, 
                batch_size=self.batch_size, 
                language=self.language, 
                previous_text=previous_text
            )

        with cuda_lock:
            result = whisperx.align(
                transcript=transcript["segments"], 
                model=self.model_a, 
                align_model_metadata=self.align_metadata, 
                audio=audio, 
                device=self.device, 
                return_char_alignments=False
            )

        return self._to_words(result)

    def _to_words(self, result):
        words = []
        for w in result["word_segments"]:
            if "start" in w and "end" in w: # ignore unaligned words
                word = Word(
                    word=w["word"],
                    start=w["start"],
                    end=w["end"],
                    sep=self.word_sep
                )
                words.append(word)
            else:
                logger.warning("Skipping unaligned word: {}", w)
        return words


class OnlineASR:
    def __init__(self, context_length=ASR_CONTEXT_LENGTH, language='en', cached=False):
        """
        context_length: number of words to condition on
        """
        self.context_length = context_length
        self.audio_buffer = None
        self.h_buffer = None
        self.asr = OfflineASR(language, cached=cached)
        
        self.reset()

    def reset(self):
        self.audio_buffer = AudioBuffer()
        self.h_buffer = HypothesisBuffer()

    @logger.catch
    def process_chunk(self, chunk, finalize=False, return_audio=False):
        sr = self.sample_rate
        
        if finalize:
            audio, buffer_offset = self.audio_buffer.clear()
            logger.debug("Flushing audio buffer of length: {:.2f}", len(audio) / sr)
            if not audio.any(): # buffer is empty
                return None
        else:
            audio, buffer_offset = self.audio_buffer.push(chunk)
            if audio is None: # buffer is not filled yet
                return None

        buffer_duration = len(audio) / sr
        buffer_end_time = buffer_offset + buffer_duration

        logger.debug("Transcribing audio of length: {:.2f}, buffer offset: {:.2f}", buffer_duration, buffer_offset)

        if self.context_length > 0:
            context = Word.to_text(self.h_buffer.confirmed_words)[-self.context_length:]
            logger.debug("Conditioning on: {}", context)
        else:
            context = None
        
        words = self.asr.transcribe(audio, previous_text=context)
        words = Word.apply_offset(words, buffer_offset)
        logger.opt(colors=True).debug("<g>Buffer transcription: {}</g>", Word.to_text(words))

        # TODO: it relays on the last word end time, which is not accurate. 
        #       better to track speech activity detection (vad)
        if words:
            silence_time = buffer_end_time - words[-1].end
        else:
            silence_time = buffer_duration
    
        confirmed_words, unconfirmed_words = self.h_buffer.update(words)

        if finalize:
            self.h_buffer.clear()
            
        if confirmed_words:
            # fast forward to the middle of the last confirmed word and the first unconfirmed word
            # it's less likely to cut the current utterance by mistake
            start = confirmed_words[-1].end
            end = unconfirmed_words[0].start if unconfirmed_words else buffer_end_time
            ff_time = (start + end) / 2
            self.audio_buffer.fast_forward(ff_time)
            logger.debug("Fast forwarding audio buffer to: {:.2f}", ff_time)

        result = {
            "confirmed_text": Word.to_text(confirmed_words),
            "unconfirmed_text": Word.to_text(unconfirmed_words),
            "silence_time": silence_time
        }

        if return_audio:
            result["audio"] = audio

        logger.debug("Confirmed text: {}", result["confirmed_text"])
        logger.debug("Unconfirmed text: {}", result["unconfirmed_text"])
        logger.debug("Silence time: {:.2f}", silence_time)

        return result

    @property
    def sample_rate(self):
        return self.audio_buffer.sampling_rate

