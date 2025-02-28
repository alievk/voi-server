import asyncio
from datetime import datetime
from loguru import logger

from llm import BaseLLMAgent
from recognition import Word, AudioBuffer, OnlineASR
from image import blob_to_openai_url


class Conversation:
    def __init__(
        self, 
        audio_input_stream,
        vad,
        asr: OnlineASR, 
        voice_generator,
        character_agent,
        conversation_context,
        stream_user_stt: bool = True,
        final_stt_correction: bool = True,
        error_cb=None,
        status_cb=None,
        event_loop=None
    ):
        self.audio_input_stream = audio_input_stream
        self.vad = vad
        self.online_asr = asr
        self.offline_asr = asr.asr
        self.user_audio_buffer = AudioBuffer(min_duration=1.0, max_duration=60)
        self.voice_generator = voice_generator
        self.character_agent = character_agent
        self.conversation_context = conversation_context
        self.stream_user_stt = stream_user_stt
        self.final_stt_correction = final_stt_correction
        self.error_cb = error_cb
        self.status_cb = status_cb
        self.user_attachments = []

        self._last_speech_text = ""
        self._stt_finished = asyncio.Event()
        self._stt_finished.set()
        self._event_loop = event_loop or asyncio.get_running_loop()

    def greeting(self):
        message = self.character_agent.greeting_message()
        if message:
            message = self.conversation_context.add_message(message)
            self._generate_voice(message)

    def on_user_audio(self, audio_chunk):
        if not self.audio_input_stream.is_running():
            self.audio_input_stream.start()
        self.audio_input_stream.put(audio_chunk)
        self._stt_finished.clear()

    @logger.catch(reraise=True)
    async def handle_input_audio(self, audio_chunk):
        """ audio_chunk is f32le """
        end_of_audio = audio_chunk is None

        STT_SILENCE_THRESHOLD = 0.2

        last_message = self.conversation_context.last_message()        

        def transcribe_audio_buffer(audio_buffer):
            if self._last_speech_text:
                cond_text = self._last_speech_text + ' '
            elif last_message and last_message["role"] == "assistant":
                cond_text = f"- {last_message['content'][0]['text']}\n- "
            else:
                cond_text = "He thoughtfuly said: "
            logger.debug(f"Conditioning ASR with: {cond_text}")
            words = self.offline_asr.transcribe(audio_buffer, cond_text)
            return Word.to_text(words)

        buffer_text = None
        if end_of_audio:
            audio_buffer, _ = self.user_audio_buffer.clear()
            buffer_text = transcribe_audio_buffer(audio_buffer)
            self.vad.reset()
        else:
            silence_duration = self.vad.process_chunk(audio_chunk)
            audio_buffer, _ = self.user_audio_buffer.push(audio_chunk)

            if self.stream_user_stt and silence_duration > STT_SILENCE_THRESHOLD and audio_buffer is not None:
                buffer_text = transcribe_audio_buffer(audio_buffer)
                self.user_audio_buffer.clear()
                self.vad.reset()

        if buffer_text:
            speech_text = self._last_speech_text + ' ' + buffer_text
            self._last_speech_text = speech_text

            if last_message and last_message["role"] == "user":
                last_message["content"][0]["text"] = speech_text
                self.conversation_context.update_message(last_message)
            else:
                message = {
                    "role": "user",
                    "content": [{"type": "text", "text": speech_text}]
                }
                self.conversation_context.add_message(message)

        if end_of_audio:
            self._last_speech_text = ""
            self._stt_finished.set()

    def on_user_text(self, text):
        message = {
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }
        self.conversation_context.add_message(message)

    def on_user_image_url(self, image_url):
        if image_url.startswith("data:image"):
            image_url = blob_to_openai_url(image_url)
        content = {
            "type": "image_url",
            "image_url": {"url": image_url}
        }
        self.user_attachments.append(content)

    def on_create_response(self):
        self.voice_generator.interrupt()
        asyncio.run_coroutine_threadsafe(self._create_response(), self._event_loop)

    async def _create_response(self):
        try:
            self.audio_input_stream.stop() # (blocking) guarantee that the user audio buffer is decoded
            await self._stt_finished.wait()
            self._attach_documents()

            messages = self.conversation_context.get_messages()
            if not messages or messages[-1]["role"] != "user":
                logger.info("The last message is not a user message, skipping response")
                self.emit_status('response_cancelled')
                return

            self.conversation_context.process_interrupted_messages()
            response = await self.character_agent.acompletion(self.conversation_context)

            if isinstance(response, dict):
                content = [{
                    "type": "text",
                    "text": response["text"]
                }]
                voice_tone = response.get("voice_tone")
            else:
                content = [{
                    "type": "text",
                    "text": response
                }]
                voice_tone = None

            message = {
                "role": "assistant",
                "content": content,
                "voice_tone": voice_tone
            }
            new_message = self.conversation_context.add_message(message)

            self._generate_voice(new_message)

            self.emit_status('response_done')
        except Exception as e:
            self.emit_error(e)
            raise e

    def _attach_documents(self):
        for content in self.user_attachments:
            last_message = self.conversation_context.last_message()
            if last_message and last_message["role"] == "user":
                last_message["content"].append(content)
                self.conversation_context.update_message(last_message)
            else:
                message = {
                    "role": "user",
                    "content": [content]
                }
                self.conversation_context.add_message(message)
        self.user_attachments = []

    def emit_status(self, status):
        if self.status_cb:
            asyncio.run_coroutine_threadsafe(self.status_cb(status), self._event_loop)

    def emit_error(self, error):
        if self.error_cb:
            asyncio.run_coroutine_threadsafe(self.error_cb(error), self._event_loop)

    def _generate_voice(self, message):
        try:
            if message.get("file"): # play cached audio
                content = f"file:{message['file']}"
            else:
                content = [c["text"] for c in message["content"] if c["type"] == "text"][0]

            if "voice_tone" in message:
                self.voice_generator.maybe_set_voice_tone(message["voice_tone"])
            self.voice_generator.generate(text=content, id=message["id"])
        except Exception as e:
            self.emit_error(e)

    def on_assistant_audio_end(self, speech_id, duration):
        messages = self.conversation_context.get_messages(filter=lambda msg: msg["role"] == "assistant" and msg["id"] == speech_id)
        assert messages, f"Message with speech_id {speech_id} not found"
        message = messages[0]
        message["audio_duration"] = duration
        self.conversation_context.update_message(message)

    def on_user_interrupt(self, speech_id, interrupted_at):
        self.voice_generator.interrupt()
        messages = self.conversation_context.get_messages(filter=lambda msg: msg["role"] == "assistant" and msg["id"] == speech_id)
        assert messages, f"Message with speech_id {speech_id} not found"
        message = messages[0]
        message["interrupted_at"] = interrupted_at
        self.conversation_context.update_message(message)