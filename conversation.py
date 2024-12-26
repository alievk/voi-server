import asyncio
from datetime import datetime
from loguru import logger

from llm import BaseLLMAgent
from recognition import OnlineASR


class Conversation:
    def __init__(
        self, 
        asr: OnlineASR, 
        voice_generator,
        character_agent,
        conversation_context,
        error_cb=None
    ):
        self.asr = asr
        self.voice_generator = voice_generator
        self.character_agent = character_agent
        self.conversation_context = conversation_context
        self.error_cb = error_cb

        self._event_loop = asyncio.get_event_loop()

    def greeting(self):
        message = self.character_agent.greeting_message()
        message = self.conversation_context.add_message(message)
        self._generate_voice(message)

    @logger.catch(reraise=True)
    async def handle_input_audio(self, audio_chunk):
        """ audio_chunk is f32le """
        end_of_audio = audio_chunk is None
        asr_result = self.asr.process_chunk(audio_chunk, finalize=end_of_audio)

        if asr_result and (asr_result["confirmed_text"] or asr_result["unconfirmed_text"]):
            message = self._create_message_from_transcription(asr_result)
            messages = self.conversation_context.get_messages()
            if messages and messages[-1]["role"] == "user":
                self.conversation_context.update_message(messages[-1]["id"], content=message["content"], handled=False)
            else:
                self.conversation_context.add_message(message)

        if end_of_audio:
            self._maybe_respond()

    def on_manual_text(self, text):
        message = {
            "role": "user",
            "content": text,
            "time": datetime.now()
        }
        self.conversation_context.add_message(message)
        self._maybe_respond()

    def _maybe_respond(self):
        asyncio.run_coroutine_threadsafe(self._maybe_respond_async(), self._event_loop)

    async def _maybe_respond_async(self):
        try:
            need_response = self.conversation_context.messages and self.conversation_context.messages[-1]["role"] == "user"
            if not need_response:
                logger.debug("No need to respond")
                return

            self.conversation_context.process_interrupted_messages()
            response = self.character_agent.completion(self.conversation_context)

            message = {
                "role": "assistant",
                "content": response,
                "time": datetime.now()
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
            elif isinstance(message["content"], dict):
                self.voice_generator.maybe_set_voice_tone(message["content"].get("voice_tone"))
                content = message["content"]["text"]
            else:
                content = message["content"]

            self.voice_generator.generate(text=content, id=message["id"])
        except Exception as e:
            self.emit_error(e)

    def _create_message_from_transcription(self, transcription):        
        return {
            "role": "user",
            "content": "..." if not transcription["confirmed_text"] else transcription["confirmed_text"],
            "time": datetime.now()
        }

    def on_assistant_audio_end(self, speech_id, duration):
        messages = self.conversation_context.get_messages(filter=lambda msg: msg["role"] == "assistant" and msg["id"] == speech_id)
        assert messages, f"Message with speech_id {speech_id} not found"
        self.conversation_context.update_message(messages[0]["id"], audio_duration=duration)

    def on_user_interrupt(self, speech_id, interrupted_at):
        messages = self.conversation_context.get_messages(filter=lambda msg: msg["role"] == "assistant" and msg["id"] == speech_id)
        assert messages, f"Message with speech_id {speech_id} not found"
        self.conversation_context.update_message(messages[0]["id"], interrupted_at=interrupted_at)