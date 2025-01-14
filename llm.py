import os
import threading
import json
from datetime import datetime
import litellm
import random
import asyncio
from loguru import logger

from text import SentenceStream


voice_tone_description = """<character voice tone> is used by the voice generator to choose the appropriate voice and intonation for <character response text>.
<character voice tone> is strictly one of the following:
  - "neutral": conversation is normal, neutral, like a business conversation or a conversation with a new acquaintance or a stranger
  - "warm": conversation is warm, like a conversation with a friend or a conversation with a partner
  - "erotic": conversation is about sex, love, or romance
  - "excited": conversation is excited, like a happy announcement or surprising news
  - "sad": conversation is sad, like a sad story or a sad conversation
"""

narrator_comment_format_description = """<character response text> contains comments made by the narrator.
The comments are always in the third person and enclosed in asterisks.
Examples:
  - Are you serious?! *her eyes widened* How are you going to do that?
  - *he looks down* I'm not sure I can do that.
  - I'm glad you're here. *she rushed to hug him*
"""

character_agent_message_format_voice_tone = (
    "Respond with the following JSON object:"
    '{"text": "<character response text>", "voice_tone": "<character voice tone>"}'
    f"\n{voice_tone_description}"
)

character_agent_message_format_narrator_comments = (
    "Respond with the following JSON object:"
    '{"text": "<character response text>", "voice_tone": "<character voice tone>"}'
    f"\n{voice_tone_description}"
    f"\n{narrator_comment_format_description}"
)

json_parse_error_response = {"text": "Sorry, I was lost in thought. Can you repeat that?", "voice_tone": "neutral"}


def stringify_content(content):
    emoji_map = lambda voice_tone: {
        "neutral": "😐",
        "warm": "😊",
        "erotic": "😍",
        "excited": "😃",
        "sad": "😔"
    }.get(voice_tone, "😐")

    if isinstance(content, dict):
        return f"{emoji_map(content.get('voice_tone'))} {content['text']}"
    else:
        return content


def model_supports_json_output(model_name):
    if "lepton" in model_name: # models provided by Lepton do not support JSON output
        return False
    return True


class ConversationContext:
    def __init__(self, messages=None, context_changed_cb=None):
        self.messages = messages or []
        self._check_messages(self.messages)
        self.lock = threading.Lock()
        self.context_changed_cb = context_changed_cb
        self._event_loop = asyncio.get_event_loop()

    def _check_messages(self, messages):
        if not messages:
            return

        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")
        if not all(isinstance(msg, dict) for msg in messages):
            raise ValueError("Messages must be a list of dictionaries")
        if "role" not in messages[0]:
            raise ValueError("Messages must be a list of dictionaries with 'role' field")
        if "content" not in messages[0]:
            raise ValueError("Messages must be a list of dictionaries with 'content' field")

    def _handle_context_changed(self):
        if self.context_changed_cb:
            asyncio.run_coroutine_threadsafe(self.context_changed_cb(self), self._event_loop)

    def add_message(self, message):
        new_message = self._add_message(message)
        self._handle_context_changed()
        return new_message

    def _add_message(self, message):
        assert isinstance(message, dict), f"Message must be a dictionary, got {message.__class__}"
        assert message["role"].lower() in ["assistant", "user"], f"Unknown role {message['role']}"
        with self.lock:
            message["id"] = len(self.messages)
            message["handled"] = False
            self.messages.append(message)
            return message

    def get_messages(self, include_fields=None, filter=None, processor=None):
        with self.lock:
            if include_fields is None:
                messages = self.messages
            else:
                messages = [{k: v for k, v in msg.items() if k in include_fields} for msg in self.messages]

            if filter:
                messages = [msg for msg in messages if filter(msg)]

            if processor:
                messages = [processor(msg) for msg in messages]

            return messages

    def update_message(self, id, **kwargs):
        with self.lock:
            for msg in self.messages:
                if msg["id"] == id:
                    msg.update(kwargs)
                    if "content" in kwargs:
                        self._handle_context_changed()
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
                        orig_content = msg["content"]["text"] if isinstance(msg["content"], dict) else msg["content"]
                        cut_content = orig_content[:int(len(orig_content) * percent)] + "... (interrupted)"
                        if isinstance(msg["content"], dict):
                            msg["content"]["text"] = cut_content
                        else:
                            msg["content"] = cut_content
                        del msg["interrupted_at"], msg["audio_duration"] # don't process this message again
                        msg["handled"] = False


@logger.catch(reraise=True)
def get_agent_config(agent_name):
    with open(os.path.join(os.path.dirname(__file__), "agents.json"), "r") as f:
        config = json.load(f)

    if agent_name not in config:
        raise ValueError(f"Agent {agent_name} not found in agents.json")

    agent_config = config[agent_name]
    for key, value in agent_config.items():
        if key.endswith("_agent"):
            agent_config[key] = get_agent_config(value)

    return agent_config


class BaseLLMAgent:
    def __init__(self, 
        model_name, 
        system_prompt, 
        examples=None
    ):
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(system_prompt)

        system_prompt = system_prompt.replace("{character_agent_message_format_voice_tone}", character_agent_message_format_voice_tone)
        system_prompt = system_prompt.replace("{character_agent_message_format_narrator_comments}", character_agent_message_format_narrator_comments)

        # force litellm to use OpenAI API if no provider is specified
        model_name = f"openai/{model_name}" if "/" not in model_name else model_name

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.examples = examples
        self._output_json = "json" in system_prompt.lower()

    @property
    def output_json(self):
        return self._output_json

    @logger.catch(reraise=True)
    def completion(self, context, stream=False, temperature=0.5):
        assert hasattr(context, 'get_messages'), "Context must have get_messages method"
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

        force_json = self.output_json and not model_supports_json_output(self.model_name)

        def message_processor(msg):
            # msg["content"] = stringify_content(msg["content"])
            if isinstance(msg["content"], dict):
                msg["content"] = json.dumps(msg["content"])

            if msg["role"] == "user" and force_json:
                msg["content"] += "\nRespond with a valid JSON object."

            return msg

        messages += context.get_messages(
            include_fields=["role", "content"], 
            processor=message_processor
        )

        if force_json and messages[-1]["role"] == "user":
            messages.append({"role": "assistant", "content": "{"}) # prefill technique https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response#example-maintaining-character-without-role-prompting

        logger.debug("LLM context:\n{}", self._messages_to_text(messages))
        
        response = litellm.completion(
            model=self.model_name, 
            messages=messages, 
            response_format={"type": "json_object"} if self.output_json else None,
            temperature=temperature,
            stream=stream
        )

        if stream:
            return SentenceStream(response, preprocessor=lambda x: x.choices[0].delta.content)

        content = response.choices[0].message.content
        logger.debug("Response content: {}", content)

        if self.output_json:
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON response: {}", content)
                content = self.llm_json_corrector(content)
                if content is None:
                    logger.error("Failed to fix JSON response, using fallback response")
                    content = json_parse_error_response

        return content

    def llm_json_corrector(self, content):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON error corrector. "
                    "You are given a JSON object that was generated by an LLM and failed to parse. "
                    "You must fix it and return a valid JSON object. "
                    "If the input is a plain text, wrap it in a JSON object according to the system prompt of the LLM:\n\n"
                    "<start of system prompt>\n"
                    f"{self.system_prompt}\n"
                    "<end of system prompt>"
                )
            },
            {
                "role": "user",
                "content": (
                    "<start of input>\n"
                    f"{content}\n"
                    "<end of input>\n"
                    "Reply only with the fixed JSON object."
                )
            },
            {
                "role": "assistant",
                "content": "{"
            }
        ]

        logger.debug("JSON corrector context:\n{}", self._messages_to_text(messages))

        response = litellm.completion(
            model=self.model_name, 
            messages=messages
        )

        content = response.choices[0].message.content

        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response: {}", content)
            return None

        logger.debug("Fixed JSON response: {}", content)
        return content

    def _extra_context_to_text(self, context):
        # Override this in child class
        return ""

    def _messages_to_text(self, messages):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])


class CharacterLLMAgent(BaseLLMAgent):
    def __init__(self, system_prompt, model_name="gpt-4o-mini", examples=None, greetings=None, control_agent=None):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt, 
            examples=examples
        )
        self.greetings = greetings
        self.control_agent = control_agent

    @staticmethod
    def from_config(config):
        control_agent = None
        if "control_agent" in config:
            if config["control_agent"]["model"] == "pattern_matching":
                control_agent = ControlPatternAgent.from_config(config["control_agent"])
            else:
                control_agent = ControlLLMAgent.from_config(config["control_agent"])

        return CharacterLLMAgent(
            system_prompt=config["system_prompt"],
            model_name=config["llm_model"],
            examples=config["examples"],
            greetings=config["greetings"],
            control_agent=control_agent
        )

    def greeting_message(self):
        greeting = random.choice(self.greetings["choices"])
        if isinstance(greeting, dict):
            content = {
                "text": greeting["content"],
                "voice_tone": self.greetings.get("voice_tone")
            }
            file = greeting["file"]
        else:
            content = greeting
            file = None

        return {
            "role": "assistant",
            "content": content,
            "file": file,
            "time": datetime.now()
        }

    def completion(self, context, stream=False, temperature=0.5):
        assert self.control_agent is None or not stream, "Control agent does not support streaming"

        if self.control_agent is None:
            return super().completion(context, stream, temperature)

        temperature_schedule = self.control_agent.temperature_schedule(temperature)
        for i_try, temperature in enumerate(temperature_schedule):
            logger.debug("Control agent try {}/{}, temperature {:.2f}", i_try + 1, len(temperature_schedule), temperature)
            response = super().completion(context, temperature=temperature)
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


class CharacterEchoAgent:
    def __init__(self, *args, **kwargs):
        pass

    def completion(self, context, stream=False, temperature=0.5):
        return context.messages[-1]["content"]

    def greeting_message(self):
        return {"role": "assistant", "content": "Hello, I'm an echo agent", "time": datetime.now()}


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

        normalize = lambda text: text.replace("'", "").replace('"', "").strip().lower()

        for phrase in self.denial_phrases:
            if normalize(phrase) in normalize(response_text):
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


def _setup_litellm():
    import dotenv
    dotenv.load_dotenv()

    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not api_base or not api_key:
        raise EnvironmentError("LITELLM_API_BASE and LITELLM_API_KEY must be defined in environment variables")

    litellm.api_base = api_base
    litellm.api_key = api_key


_setup_litellm()