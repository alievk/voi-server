import threading
import json
from datetime import datetime
import litellm
import random
from loguru import logger

from text import SentenceStream


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
        self._output_json = "json" in system_prompt.lower()

    @property
    def output_json(self):
        return self._output_json

    def completion(self, context, stream=False, temperature=0.5):
        assert isinstance(context, ConversationContext), f"Invalid context type {context.__class__}"
        assert not (stream and self.output_json), "Streamed JSON responses are not supported"
        
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
            temperature=temperature,
            stream=stream
        )

        if not stream:
            content = response.choices[0].message.content
            logger.debug("Response content: {}", content)

            if self.output_json:
                content = json.loads(content)

            return content

        return SentenceStream(response, preprocessor=lambda x: x.choices[0].delta.content)

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
    def __init__(self, system_prompt, model_name="gpt-4o-mini", examples=None, greetings=None, control_agent=None):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt, 
            examples=examples
        )
        self.greetings = greetings
        self.control_agent = control_agent

    def greeting_message(self):
        return {
            "role": "assistant",
            "content": self.greetings[random.randint(0, len(self.greetings) - 1)],
            "time": datetime.now()
        }

    def completion(self, context, stream=False, temperature=0.5):
        assert self.control_agent is None or not stream, "Control agent does not support streaming"

        if self.control_agent is None:
            return super().completion(context, stream, temperature)

        temperature_schedule = self.control_agent.temperature_schedule(temperature)
        for i_try, temperature in enumerate(temperature_schedule):
            logger.debug("Control agent try {}/{}, temperature {:.2f}", i_try + 1, len(temperature_schedule), temperature)
            response = super().completion(context, stream=False, temperature=temperature)
            if self.control_agent.classify(response):
                return response
            logger.debug("Control agent denied response, increasing temperature")

        logger.warning("Control agent failed after {} tries", len(temperature_schedule))
        if self.control_agent.giveup_response:
            giveup_response = random.choice(self.control_agent.giveup_response)
            if isinstance(response, dict):
                response["text"] = giveup_response
            else:
                response = giveup_response
            logger.debug("Control agent giving up, using fallback response: {}", giveup_response)
        return response


class ControlBaseAgent:
    def __init__(self, temperature_multiplier=None, giveup_after=None, giveup_response=None):
        self.temperature_multiplier = 1.5 if temperature_multiplier is None else temperature_multiplier
        self.giveup_after = 3 if giveup_after is None else giveup_after
        self.giveup_response = giveup_response

    def temperature_schedule(self, temperature_0):
        return [temperature_0 * self.temperature_multiplier ** i for i in range(self.giveup_after)]

    def classify(self, response):
        raise NotImplementedError


class ControlPatternAgent(ControlBaseAgent):
    def __init__(self, denial_phrases, temperature_multiplier=None, giveup_after=None, giveup_response=None):
        super().__init__(temperature_multiplier, giveup_after, giveup_response)
        self.denial_phrases = denial_phrases

    @staticmethod
    def from_config(config):
        return ControlPatternAgent(
            denial_phrases=config["denial_phrases"],
            temperature_multiplier=config.get("temperature_multiplier"),
            giveup_after=config.get("giveup_after"),
            giveup_response=config.get("giveup_response")
        )

    def classify(self, response):
        if isinstance(response, dict):
            response_text = response["text"]
        else:
            response_text = response

        for phrase in self.denial_phrases:
            if phrase.lower() in response_text.lower():
                return False

        return True


class ControlLLMAgent(BaseLLMAgent):
    def __init__(self, system_prompt, denial_classes: list[str], model_name="gpt-4o-mini", examples=None, temperature_multiplier=None, giveup_after=None, giveup_response=None):
        super().__init__(
            model_name=model_name, 
            system_prompt=system_prompt, 
            examples=examples,
            temperature_multiplier=temperature_multiplier,
            giveup_after=giveup_after,
            giveup_response=giveup_response
        )
        assert self.output_json, "Control agent must be used with JSON responses"
        self.denial_classes = denial_classes

    @staticmethod
    def from_config(config):
        return ControlLLMAgent(
            system_prompt=config["system_prompt"],
            denial_classes=config["denial_classes"],
            model_name=config.get("model"),
            examples=config.get("examples"),
            temperature_multiplier=config.get("temperature_multiplier"),
            giveup_after=config.get("giveup_after"),
            giveup_response=config.get("giveup_response")
        )

    def classify(self, response):
        if isinstance(response, dict):
            response_text = response["text"]
        else:
            response_text = response

        context = ConversationContext()
        context.add_message({"role": "user", "content": response_text})
        control_response = super().completion(context)

        if control_response["class"] in self.denial_classes:
            return False
        
        return True

