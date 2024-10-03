import numpy as np
import librosa
import soundfile
import time
import torch
from loguru import logger

import whisperx


SAMPLING_RATE = 16000

MIN_AUDIO_BUFFER_SIZE = 3
MAX_AUDIO_BUFFER_SIZE = 15

ASR_CONTEXT_LENGTH = 200


def load_audio(fname, sr):
    a, _ = librosa.load(fname, sr=sr, dtype=np.float32)
    return a


class FakeAudioStream:
    def __init__(self, filename, sr=SAMPLING_RATE):
        self.audio = load_audio(filename, sr)
        self.duration = len(self.audio) / sr
        self.sr = sr
        self.beg = 0

        logger.debug("Loaded audio is {:.2f} seconds", self.duration)

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


class AudioBuffer:
    def __init__(self, sampling_rate=SAMPLING_RATE, min_size=MIN_AUDIO_BUFFER_SIZE, max_size=MAX_AUDIO_BUFFER_SIZE):
        self.sampling_rate = sampling_rate
        self.min_size = int(min_size * sampling_rate)
        self.max_size = int(max_size * sampling_rate)
        self.buffer = None
        self.offset = None # global time of the buffer's first element
        self.clear()

    def push(self, data):
        assert isinstance(data, np.ndarray), f"Invalid data type: {type(data)}"
        self.buffer = np.concatenate([self.buffer, data])
        if len(self.buffer) < self.min_size:
            logger.debug("Audio buffer is shorter than {}s", MIN_AUDIO_BUFFER_SIZE)
            return None, None
        elif len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
            self.offset += (len(self.buffer) - self.max_size) / self.sampling_rate
        return self.buffer, self.offset

    def clear(self):
        buffer, offset = self.buffer, self.offset
        self.buffer = np.empty((0,), dtype=np.float32)
        self.offset = 0
        return buffer, offset

    def fast_forward(self, time):
        assert time >= self.offset, f"Negative fast forward unsupported ({time=}, {self.offset=})"
        ff = min(time - self.offset, len(self.buffer) / self.sampling_rate)
        ff_size = int(ff * self.sampling_rate)
        self.buffer = self.buffer[ff_size:]
        self.offset += ff

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
            #logger.debug("before fast forward:")
            #logger.debug("unconf: {}", Word.to_text(self.unconfirmed_words))
            #logger.debug("currnt: {}", Word.to_text(current_words))
            current_words = self._fast_forward(current_words, self.confirmed_words[-1].end)
            self.unconfirmed_words = self._fast_forward(self.unconfirmed_words, self.confirmed_words[-1].end)
            #logger.debug("after fast forward:")
            #logger.debug("unconf: {}", Word.to_text(self.unconfirmed_words))
            #logger.debug("currnt: {}", Word.to_text(current_words))
            
        if self.unconfirmed_words:
            lcp = self._longest_common_prefix(self.unconfirmed_words, current_words)
            self.confirmed_words.extend(lcp)

            if lcp:
                current_words = self._fast_forward(current_words, self.confirmed_words[-1].end)

        self.unconfirmed_words = current_words

    @staticmethod
    def _longest_common_prefix(s1, s2):
        min_len = min(len(s1), len(s2))
        logger.debug("comparing:")
        logger.debug("unconf: {}", Word.to_text(s1))
        logger.debug("currnt: {}", Word.to_text(s2))
        
        index = 0
        while index < min_len and s1[index].word.lower() == s2[index].word.lower():
            index += 1

        return s1[:index]

    @staticmethod
    def _fast_forward(words, time):
        # fast forward `words` till the first word which starts after `time`
        while words and words[0].start < time:
            words.pop(0)
        return words


class OfflineASR:
    word_sep = " "
    
    def __init__(self, language):
        self.device = "cuda"
        self.batch_size = 1 # reduce if low on GPU mem
        self.language = language

        self.model = whisperx.load_model(
            "large-v2", 
            self.device, 
            compute_type="float16", 
            language=self.language,
            # vad_options={'vad_onset': 0.8, 'vad_offset': 0.8}
            # asr_options={"initial_prompt": "He thoughtfully said: "}
        )
        
        self.model_a, self.align_metadata = whisperx.load_align_model(
            language_code=self.language, 
            device=self.device, 
            model_name="WAV2VEC2_ASR_LARGE_LV60K_960H"
        )

    def transcribe(self, audio, previous_text=None):
        result = self.model.transcribe(audio, batch_size=self.batch_size, language=self.language, previous_text=previous_text)
        result = whisperx.align(result["segments"], self.model_a, self.align_metadata, audio, self.device, return_char_alignments=False)
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
    def __init__(self):
        self.audio_buffer = None
        self.h_buffer = None
        self.asr = OfflineASR('en')
        
        self.reset()

    def reset(self):
        self.audio_buffer = AudioBuffer()
        self.h_buffer = HypothesisBuffer()

    @logger.catch
    def process_chunk(self, chunk, finalize=False, return_audio=False):
        if finalize:
            audio, offset = self.audio_buffer.clear()
            logger.debug("Flushing audio buffer of length: {:.2f}", len(audio) / SAMPLING_RATE)
            if not audio.any(): # buffer is empty
                return None
        else:
            audio, offset = self.audio_buffer.push(chunk)
            if audio is None: # buffer is not filled yet
                return None

        logger.debug("Transcribing audio of length: {:.2f}, offset: {:.2f}", len(audio) / SAMPLING_RATE, offset)

        confirmed_text = Word.to_text(self.h_buffer.confirmed_words)[-ASR_CONTEXT_LENGTH:]
        logger.debug("Conditioning on: {}", confirmed_text)
        
        words = self.asr.transcribe(audio, previous_text=confirmed_text)
        logger.opt(colors=True).debug("<g>Buffer transcription: {}</g>", Word.to_text(words))
    
        words = Word.apply_offset(words, offset)
    
        self.h_buffer.update(words)
        
        confirmed_words = self.h_buffer.confirmed_words
        if confirmed_words:
            ff_time = confirmed_words[-1].end
            logger.debug("Fast forwarding audio buffer to: {:.2f}", ff_time)
            self.audio_buffer.fast_forward(ff_time)

        result = {
            "confirmed_text": Word.to_text(confirmed_words),
            "unconfirmed_text": Word.to_text(self.h_buffer.unconfirmed_words)
        }

        if return_audio:
            result["audio"] = audio

        logger.debug("Confirmed text: {}", result["confirmed_text"])
        logger.debug("Unconfirmed text: {}", result["unconfirmed_text"])

        return result

