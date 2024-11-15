import threading
import json
from datetime import datetime
import litellm
import random
from loguru import logger


litellm.api_base = "http://13.43.85.180:4000"
litellm.api_key = "sk-1234"


user_message_format_description = """User messages are transcriptions of the user's audio.
Transcriptions consist of confirmed words followed by unconfirmed words in parentheses.
Confirmed words are the words reliably recognized by the speech-to-text system.
Unconfirmed words are the words which are not reliably recognized.
You must ignore unconfirmed words if:
  - They are not consistent with the user's previous speech
  """

agent_message_format_description = """Respond with the following JSON object:
{"text": "<agent response text>", "voice_tone": "<voice tone>"}
<voice_tone> is used by the voice generator to choose the appropriate voice and intonation for <agent response text>.
<voice_tone> is strictly one of the following:
  - "neutral": the voice tone is normal, neutral
  - "passionate": the voice tone is passionate, like a declaration of love or a passionate conversation
  - "excited": the voice tone is excited, like a happy announcement or an excited conversation
  - "sad": the voice tone is sad, like a sad story or a sad conversation
"""


class ConversationContext:
    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()

    def add_message(self, message, text_compare_f=None):
        with self.lock:
            assert message["role"].lower() in ["assistant", "user"], f"Unknown role {message['role']}"

            if not self.messages or self.messages[-1]["role"].lower() != message["role"].lower():
                message["id"] = len(self.messages)
                message["handled"] = False
                self.messages.append(message)
                return True, message

            if text_compare_f is None:
                text_compare_f = lambda x, y: x == y

            if not text_compare_f(self.messages[-1]["content"], message["content"]):
                self.messages[-1]["content"] = message["content"]
                self.messages[-1]["handled"] = False
                return True, self.messages[-1]

            return False, None

    def get_messages(self, include_fields=None, filter=None):
        with self.lock:
            if include_fields is None:
                messages = self.messages
            else:
                messages = [{k: v for k, v in msg.items() if k in include_fields} for msg in self.messages]

            if filter:
                messages = [msg for msg in messages if filter(msg)]

            return messages

    def update_message(self, id, **kwargs):
        with self.lock:
            for msg in self.messages:
                if msg["id"] == id:
                    msg.update(kwargs)
                    return True
            return False

    def to_text(self):
        with self.lock:
            if not self.messages:
                return ""

            lines = []
            for item in self.messages:
                if not "time" in item:
                    time_str = "00:00:00"
                else:
                    time_str = item["time"].strftime("%H:%M:%S")
                
                role = item["role"].lower()
                content = item["content"]
                
                lines.append(f"{time_str} | {role} - {content}")
            
            return "\n".join(lines)

    def process_interrupted_messages(self):
        # FIXME: simple and dirty way to process interrupted messages
        with self.lock:
            for msg in self.messages:
                if "interrupted_at" in msg and "audio_duration" in msg:
                    # Cut the message content to the point where it was interrupted
                    percent = msg["interrupted_at"] / msg["audio_duration"]
                    if percent < 1: # if percent > 1, the message was not interrupted
                        cut_content = msg["content"][:int(len(msg["content"]) * percent)]
                        msg["content"] = f"{cut_content}... (interrupted)"
                        del msg["interrupted_at"], msg["audio_duration"] # don't process this message again
                        msg["handled"] = False


class BaseLLMAgent:
    def __init__(self, model_name, system_prompt, examples=None):
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(system_prompt)

        system_prompt = system_prompt.replace("{user_message_format_description}", user_message_format_description)
        system_prompt = system_prompt.replace("{agent_message_format_description}", agent_message_format_description)

        if not model_name.startswith("openai"): # force litellm to use OpenAI API
            model_name = f"openai/{model_name}"

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.examples = examples
        self.output_json = "json" in system_prompt.lower()

    def completion(self, context):
        assert isinstance(context, ConversationContext), f"Invalid context type {context.__class__}"
        
        messages = [
            {
              "role": "system",
              "content": self.system_prompt
            }
        ]

        if self.examples:
            for example in self.examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

        messages += context.get_messages(include_fields=["role", "content"])

        logger.debug("Messages:\n{}", self._messages_to_text(messages))
        
        response = litellm.completion(
            model=self.model_name, 
            messages=messages, 
            response_format={"type": "json_object"} if self.output_json else None,
            temperature=0.5
        )

        content = response.choices[0].message.content
        logger.debug("Response content: {}", content)

        if self.output_json:
            content = json.loads(content)

        return {
            "content": content,
            "messages": messages
        }

    @staticmethod
    def format_transcription(confirmed, unconfirmed):
        text = confirmed
        if unconfirmed:
            text += f" ({unconfirmed})"
        return text.strip()

    def _extra_context_to_text(self, context):
        # Override this in child class
        return ""

    def _messages_to_text(self, messages):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])


class ResponseLLMAgent(BaseLLMAgent):
    default_content = ""
    
    def __init__(self, system_prompt, model_name="gpt-4o-mini", examples=None, greetings=None):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt, 
            examples=examples
        )
        self.greetings = greetings

    def completion(self, context, *args, **kwargs):
        result = super().completion(context, *args, **kwargs)

        if result["content"] is None:
            result["content"] = self.default_content

        return result

    def greeting_message(self):
        return {
            "role": "assistant",
            "content": self.greetings[random.randint(0, len(self.greetings) - 1)],
            "time": datetime.now()
        }
