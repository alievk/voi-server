import asyncio
from datetime import datetime
from loguru import logger

from llm import BaseLLMAgent
from recognition import Word, AudioBuffer, OnlineASR
from image import image_to_openai_url


class Conversation:
    def __init__(
        self, 
        asr: OnlineASR, 
        voice_generator,
        character_agent,
        conversation_context,
        stream_user_stt: bool = True,
        final_stt_correction: bool = True,
        error_cb=None
    ):
        self.online_asr = asr
        self.offline_asr = asr.asr
        self.user_audio_buffer = AudioBuffer(min_duration=0.1, max_duration=60)
        self.voice_generator = voice_generator
        self.character_agent = character_agent
        self.conversation_context = conversation_context
        self.stream_user_stt = stream_user_stt
        self.final_stt_correction = final_stt_correction
        self.error_cb = error_cb

        self._event_loop = asyncio.get_event_loop()

        self.attachments = []

    def greeting(self):
        message = self.character_agent.greeting_message()
        if message:
            message = self.conversation_context.add_message(message)
            self._generate_voice(message)

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
            messages = self.conversation_context.get_messages()
            if messages and messages[-1]["role"] == "user":
                message = messages[-1]
                for content in message["content"]:
                    if content["type"] == "text":
                        content["text"] = text
                message["handled"] = False
                self.conversation_context.update_message(message)
            else:
                message = {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                    "time": datetime.now(),
                    "from": "audio"
                }
                self.conversation_context.add_message(message)

        if end_of_audio:
            self._maybe_respond()

    def on_manual_text(self, text):
        content = [{"type": "text", "text": text}]
        if self.attachments:
            content.extend(self.attachments)
            self.attachments = []

        message = {
            "role": "user",
            "content": content,
            "time": datetime.now(),
            "from": "text"
        }
        self.conversation_context.add_message(message)
        self._maybe_respond()

    def on_image_blob(self, image):
        self.attachments.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_openai_url(image)}
            }
        )

    def on_image_url(self, image_url):
        self.attachments.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        )

    def _maybe_respond(self):
        asyncio.run_coroutine_threadsafe(self._maybe_respond_async(), self._event_loop)

    async def _maybe_respond_async(self):
        try:
            messages = self.conversation_context.get_messages()
            if not messages or messages[-1]["role"] != "user":
                logger.debug("No need to respond")
                return

            if self.attachments:
                # last_message["content"].extend(self.attachments) # not supported yet
                self.attachments = []

            self.conversation_context.process_interrupted_messages()
            response = self.character_agent.completion(self.conversation_context)

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
                "voice_tone": voice_tone,
                "time": datetime.now(),
                "from": "llm"
            }
            new_message = self.conversation_context.add_message(message)

            self._generate_voice(new_message)
        except Exception as e:
            self.emit_error(e)
            raise e

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
        messages = self.conversation_context.get_messages(filter=lambda msg: msg["role"] == "assistant" and msg["id"] == speech_id)
        assert messages, f"Message with speech_id {speech_id} not found"
        message = messages[0]
        message["interrupted_at"] = interrupted_at
        self.conversation_context.update_message(message)