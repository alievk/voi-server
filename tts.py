import asyncio
import queue
import threading

from TTS.api import TTS


# OK speakers: 43, 49, 56
SPEAKERS = ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']


class VoiceGenerator:
    _cached_tts_model = None

    def __init__(self, generated_audio_cb=None, speaker=SPEAKERS[49], cached=False):
        """ generated_audio_cb receives f32le audio chunks """
        self.generated_audio_cb = generated_audio_cb
        self.speaker = speaker
        self.language = 'en'
        self.tts = VoiceGenerator.get_model(cached=cached)

        self.running = False
        self.text_queue = None
        self._event_loop = asyncio.get_event_loop()

    def start(self):
        self.running = True
        self.text_queue = queue.Queue()
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.start()

    @staticmethod
    def get_model(cached=True):
        if cached and VoiceGenerator._cached_tts_model is not None:
            return VoiceGenerator._cached_tts_model

        VoiceGenerator._cached_tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        return VoiceGenerator._cached_tts_model

    def generate(self, text, streaming=False):
        if streaming:
            return self._stream_generator(text)

        return self.tts.tts(text=text, speaker=self.speaker, language=self.language)

    def generate_async(self, text):
        if not self.running:
            raise RuntimeError("VoiceGenerator is not running")

        self.text_queue.put(text)

    def _processing_loop(self):
        while self.running:
            text = self.text_queue.get()
            if text is None:
                break

            for chunk in self.generate(text, streaming=True):
                asyncio.run_coroutine_threadsafe(self.generated_audio_cb(chunk), self._event_loop)
                if not self.running or not self.text_queue.empty(): # stop on new text or interrupt
                    break

            # end of generation
            asyncio.run_coroutine_threadsafe(self.generated_audio_cb(None), self._event_loop)

    def _stream_generator(
        self,
        text: str
    ):
        latents = self.tts.synthesizer.tts_model.speaker_manager.speakers[self.speaker]
        chunks = self.tts.synthesizer.tts_model.inference_stream(
            text=text,
            language=self.language,
            gpt_cond_latent=latents["gpt_cond_latent"],
            speaker_embedding=latents["speaker_embedding"]
        )

        for chunk in chunks:
            yield chunk.cpu().numpy()

    @property
    def sample_rate(self):
        return self.tts.synthesizer.output_sample_rate

    def stop(self):
        self.running = False
        self.text_queue.put(None)
        self.thread.join(timeout=5)
        self.text_queue = None
