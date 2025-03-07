import asyncio
from datetime import datetime
from loguru import logger
from typing import Callable

from llm import ConversationContext, CharacterLLMAgent, CompletenessAgent, AgentConfigManager
from recognition import ASRWithVAD
from generation import AsyncVoiceGenerator
from image import blob_to_openai_url
from audio import AudioInputStream
from utils import Once


call_mode_params = {
    "user_silence_before_response": {
        "complete": 0,
        "incomplete": 2
    }
}


class Conversation:
    def __init__(
        self, 
        audio_input_stream: AudioInputStream,
        asr: ASRWithVAD,
        voice_generator: AsyncVoiceGenerator,
        character_agent: CharacterLLMAgent,
        conversation_context: ConversationContext,
        mode: str = "chat",
        stream_user_stt: bool = True,
        final_stt_correction: bool = True,
        error_cb: Callable = None,
        status_cb: Callable = None,
        event_loop: asyncio.AbstractEventLoop = None
    ):
        self.audio_input_stream = audio_input_stream
        self.asr = asr
        self.voice_generator = voice_generator
        self.character_agent = character_agent
        self.conversation_context = conversation_context
        self.mode = mode
        self.stream_user_stt = stream_user_stt
        self.final_stt_correction = final_stt_correction
        self.error_cb = error_cb
        self.status_cb = status_cb
        self.user_attachments = []

        self._stt_finished = asyncio.Event()
        self._stt_finished.set()
        self._event_loop = event_loop or asyncio.get_running_loop()

        self._last_request_hash = None
        self._user_message_status = "incomplete"
        self._interrupt_once = Once()

        agent_config_manager = AgentConfigManager()
        agent_config = agent_config_manager.get_config("completeness_agent")
        self._completeness_agent = CompletenessAgent.from_config(agent_config)

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
        asr_result = self.asr.process_chunk(audio_chunk, context=self.conversation_context)

        if self.mode == "call" and asr_result["vad_stats"]["trailing_voice"] > 0:
            self._interrupt_once.call(self.on_user_interrupt)

        if asr_result["text"]:
            logger.debug(f"Adding text chunk: {asr_result['text']}")
            last_message = self.conversation_context.last_message()
            if last_message and last_message["role"] == "user":
                last_message["content"][0]["text"] += ' ' + asr_result["text"]
                self.conversation_context.update_message(last_message)
            else:
                message = {
                    "role": "user",
                    "content": [{"type": "text", "text": asr_result["text"]}]
                }
                self.conversation_context.add_message(message)

            # if self.mode == "call":
            #     asyncio.run_coroutine_threadsafe(
            #         self._update_context(),
            #         self._event_loop
            #     )

        if audio_chunk is None: # end of user audio
            self._stt_finished.set()

        context_hash = self.conversation_context.get_hash()
        trailing_silence = asr_result["vad_stats"]["trailing_silence"]
        user_silence_before_response = call_mode_params["user_silence_before_response"][self._user_message_status]
        if (
            self.mode == "call" and 
            trailing_silence > user_silence_before_response and
            self._last_request_hash != context_hash
        ):
            logger.info(f"Triggering response for trailing silence {trailing_silence} > {user_silence_before_response}")
            self.on_create_response()
            self._interrupt_once.reset()
            self._last_request_hash = context_hash

    @logger.catch(reraise=True)
    async def _update_context(self):
        response = await self._completeness_agent.aclassify(self.conversation_context)
        self._user_message_status = response["status"]
        logger.debug(f"User message status: {self._user_message_status}")

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
        asyncio.run_coroutine_threadsafe(self._create_response(), self._event_loop)

    async def _create_response(self):
        try:
            messages = self.conversation_context.get_messages()
            if not messages or messages[-1]["role"] != "user":
                logger.info("The last message is not a user message, skipping response")
                self.emit_status('response_cancelled')
                return

            self.audio_input_stream.stop() # (blocking) guarantee that the user audio buffer is decoded
            await self._stt_finished.wait()
            self._attach_documents()

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

    def on_user_interrupt(self, speech_id=None, interrupted_at=None):
        self.voice_generator.interrupt()
        
        if self.mode == "call":
            logger.info(f"Triggering interrupt for detected user voice.")
            self.emit_status("user_interrupt")

        if speech_id:
            messages = self.conversation_context.get_messages(filter=lambda msg: msg["role"] == "assistant" and msg["id"] == speech_id)
            assert messages, f"Message with speech_id {speech_id} not found"
            message = messages[0]
            message["interrupted_at"] = interrupted_at
            self.conversation_context.update_message(message)