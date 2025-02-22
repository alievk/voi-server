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
        asr: OnlineASR, 
        voice_generator,
        character_agent,
        conversation_context,
        stream_user_stt: bool = True,
        final_stt_correction: bool = True,
        error_cb=None,
        event_loop=None
    ):
        self.audio_input_stream = audio_input_stream
        self.online_asr = asr
        self.offline_asr = asr.asr
        self.user_audio_buffer = AudioBuffer(min_duration=0.1, max_duration=60)
        self.voice_generator = voice_generator
        self.character_agent = character_agent
        self.conversation_context = conversation_context
        self.stream_user_stt = stream_user_stt
        self.final_stt_correction = final_stt_correction
        self.error_cb = error_cb
        self.user_attachments = []

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
        asr_result = None

        if end_of_audio:
            if self.stream_user_stt:
                # we need to finalize the online asr even if we don't use the result
                asr_result = self.online_asr.process_chunk(audio_chunk, finalize=True)

            if self.final_stt_correction and not self.user_audio_buffer.empty():
                logger.debug("Performing final stt correction")
                words = self.offline_asr.transcribe(self.user_audio_buffer.buffer)
                text = Word.to_text(words)

                logger.debug(f"Final stt correction: {text}")
                asr_result = {
                    "confirmed_text": text,
                    "unconfirmed_text": "",
                }

            self.user_audio_buffer.clear()
        else:
            if self.stream_user_stt:
                asr_result = self.online_asr.process_chunk(audio_chunk)

            self.user_audio_buffer.push(audio_chunk)

        if asr_result and (asr_result["confirmed_text"] or asr_result["unconfirmed_text"]):
            text = asr_result["confirmed_text"] or asr_result["unconfirmed_text"] or "..."
            last_message = self.conversation_context.last_message()
            if last_message and last_message["role"] == "user":
                last_message["content"][0]["text"] = text
                self.conversation_context.update_message(last_message)
            else:
                message = {
                    "role": "user",
                    "content": [{"type": "text", "text": text}]
                }
                self.conversation_context.add_message(message)

        if end_of_audio:
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
                logger.debug("No need to respond")
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