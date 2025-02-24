import os
import threading
import json
from datetime import datetime
import litellm
import random
import asyncio
import copy
from loguru import logger

from text import SentenceStream, NARRATOR_MARKER


voice_tone_description = """<character voice tone> is used by the voice generator to choose the appropriate voice and intonation for <character response text>.
<character voice tone> is strictly one of the following:
  - "neutral": conversation is normal, neutral, like a business conversation or a conversation with a new acquaintance or a stranger
  - "warm": conversation is warm, like a conversation with a friend or a conversation with a partner
  - "erotic": conversation is about sex, love, or romance
  - "excited": conversation is excited, like a happy announcement or surprising news
  - "sad": conversation is sad, like a sad story or a sad conversation
"""

narrator_comment_format_description = f"""<character response text> contains comments made by the narrator.
The comments are always in the third person and enclosed in {NARRATOR_MARKER}.
Examples:
  - Are you serious?! {NARRATOR_MARKER}her eyes widened{NARRATOR_MARKER} How are you going to do that?
  - {NARRATOR_MARKER}he looks down{NARRATOR_MARKER} I'm not sure I can do that.
  - I'm glad you're here. {NARRATOR_MARKER}she rushed to hug him{NARRATOR_MARKER}
"""

prompt_patterns = {
    "character_agent_message_format_voice_tone": (
        "Respond with the following JSON object:"
        '{"text": "<character response text>", "voice_tone": "<character voice tone>"}'
        f"\n{voice_tone_description}"
    ),

    "character_agent_message_format_narrator_comments": (
        "Respond with the following JSON object:"
        '{"text": "<character response text>", "voice_tone": "<character voice tone>"}'
        f"\n{voice_tone_description}"
        f"\n{narrator_comment_format_description}"
    )
}

json_parse_error_response = {"text": "Sorry, I was lost in thought. Can you repeat that?", "voice_tone": "neutral"}


def voice_tone_emoji(voice_tone):
    return {
        "neutral": "😐",
        "warm": "😊",
        "erotic": "😍",
        "excited": "😃",
        "sad": "😔"
    }.get(voice_tone, "😐")


def model_supports_json_output(model_name):
    if "lepton" in model_name: # models provided by Lepton do not support JSON output
        return False
    return True


class ConversationContext:
    def __init__(self, context_changed_cb=None, event_loop=None):
        self.context_changed_cb = context_changed_cb
        self.lock = threading.Lock()
        self._event_loop = event_loop or asyncio.get_running_loop()
        self._messages = []

    def _handle_context_changed(self, message):
        if self.context_changed_cb:
            asyncio.run_coroutine_threadsafe(self.context_changed_cb(message), self._event_loop)

    def _check_message(self, message):
        # Check important fields
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
        if "role" not in message:
            raise ValueError("Message must have 'role' field")
        if message["role"].lower() not in ["assistant", "user"]:
            raise ValueError(f"Unknown role {message['role']}")
        if "content" not in message:
            raise ValueError("Message must have 'content' field")
        if not isinstance(message["content"], list):
            raise ValueError("'content' field must be a list")
        
        def check_content(d):
            if not isinstance(d, dict):
                raise ValueError("Content must be a dictionary")
            if "type" not in d:
                raise ValueError("Content must have 'type' field")
            if d["type"] == "text":
                if "text" not in d:
                    raise ValueError("Text must have 'text' field")
                if not isinstance(d["text"], str):
                    raise ValueError("Text must be a string")
            elif d["type"] == "image_url":
                if "image_url" not in d:
                    raise ValueError("Image URL must have 'image_url' field")
                if not isinstance(d["image_url"], dict):
                    raise ValueError("Image URL must be a dictionary")
                if "url" not in d["image_url"]:
                    raise ValueError("Image URL must have 'url' field")
                if not isinstance(d["image_url"]["url"], str):
                    raise ValueError("Image URL must be a string")
            else:
                raise ValueError(f"Unknown content type {d['type']}")

        for content in message["content"]:
            check_content(content)

    def add_message(self, message):
        new_message = self._add_message(message)
        self._handle_context_changed(new_message)
        return new_message

    def _add_message(self, message):
        try:
            self._check_message(message)
        except ValueError as e:
            logger.error("Invalid message: {}", message)
            raise e
        message["id"] = len(self._messages)
        with self.lock:
            self._messages.append(message)
            return message

    def get_messages(self, include_fields=None, filter=None, processor=None):
        with self.lock:
            if include_fields is None:
                messages = copy.deepcopy(self._messages)
            else:
                messages = [{k: v for k, v in msg.items() if k in include_fields} for msg in self._messages]

            if filter:
                messages = [msg for msg in messages if filter(msg)]

            if processor:
                messages = [processor(msg) for msg in messages]

            return messages

    def last_message(self):
        with self.lock:
            if not self._messages:
                return None
            return copy.deepcopy(self._messages[-1])

    def update_message(self, message):
        self._check_message(message)
        assert "id" in message, "Message must have 'id' field"
        with self.lock:
            for msg in self._messages:
                if msg["id"] == message["id"]:
                    context_changed = str(msg["content"]) != str(message["content"])
                    msg.update(message)
                    if context_changed:
                        self._handle_context_changed(msg)
                    return True
            return False

    def process_interrupted_messages(self):
        # FIXME: simple and dirty way to process interrupted messages
        with self.lock:
            for msg in self._messages:
                if "interrupted_at" in msg and "audio_duration" in msg:
                    # Cut the message content to the point where it was interrupted
                    percent = msg["interrupted_at"] / msg["audio_duration"]
                    if percent < 1: # if percent > 1, the message was not interrupted
                        orig_content = msg["content"][0]["text"]
                        msg["content"][0]["text"] = orig_content[:int(len(orig_content) * percent)] + "... (interrupted)"
                        del msg["interrupted_at"], msg["audio_duration"] # don't process this message again
                        self._handle_context_changed(msg)


class AgentConfigManager:
    def __init__(self):
        self._agent_list = self._load_agent_list()
        self._resolve_refs()

    def _load_agent_list(self):
        agent_list = {}
        for file in os.listdir(os.path.join(os.path.dirname(__file__), "agents")):
            if file.endswith(".json"):
                with open(os.path.join(os.path.dirname(__file__), "agents", file), "r") as f:
                    agent_list.update(json.load(f))
        return agent_list

    def _resolve_refs(self):
        # resolve nested agents
        for agent_name, agent_config in self._agent_list.items():
            for key, value in agent_config.items():
                if key.endswith("_agent") and isinstance(value, str):
                    self._agent_list[agent_name][key] = self._agent_list[value]

        def resolve(value, vars=None):
            if isinstance(value, list):
                return [resolve(v, vars) for v in value]
            elif isinstance(value, dict) and "content" in value:
                return resolve(value["content"], vars)

            for pattern_name, pattern in prompt_patterns.items():
                value = value.replace(f"{{prompt_pattern:{pattern_name}}}", pattern)

            if vars:
                for k, v in vars.items():
                    value = value.replace(f"{{vars:{k}}}", str(v))

            return value

        for c in self._agent_list.values():
            if "system_prompt" in c:
                c["system_prompt"] = resolve(c["system_prompt"], c.get("vars"))
            if "examples" in c:
                c["examples"] = resolve(c["examples"], c.get("vars"))
            if "greetings" in c:
                c["greetings"]["choices"] = resolve(c["greetings"]["choices"], c.get("vars"))

    def add_agent(self, agent_name, agent_config):
        self._agent_list[agent_name] = agent_config
        self._resolve_refs()

    def get_config(self, agent_name):
        if agent_name not in self._agent_list:
            raise ValueError(f"Agent {agent_name} not found")
        return self._agent_list[agent_name]


class BaseLLMAgent:
    def __init__(self, 
        model_name, 
        system_prompt, 
        examples=None,
        event_loop=None
    ):
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(system_prompt)

        # force litellm to use OpenAI API if no provider is specified
        model_name = f"openai/{model_name}" if "/" not in model_name else model_name

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.examples = examples
        self._output_json = "json" in system_prompt.lower()
        self._event_loop = event_loop or asyncio.get_running_loop()

    @property
    def output_json(self):
        return self._output_json

    @logger.catch(reraise=True)
    async def acompletion(self, context, stream=False, temperature=0.5):
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
            if msg["role"] == "user" and force_json:
                for content in msg["content"]:
                    if content["type"] == "text":
                        content["text"] = f"{content['text']}\nRespond with a valid JSON object."
            elif msg["role"] == "assistant":
                msg["content"] = msg["content"][0]["text"]
            return msg

        messages += context.get_messages(
            include_fields=["role", "content"], 
            processor=message_processor
        )

        if force_json and messages[-1]["role"] == "user":
            messages.append({"role": "assistant", "content": "{"}) # prefill technique https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response#example-maintaining-character-without-role-prompting

        logger.debug("LLM context:\n{}", self._messages_to_text(messages))
        
        response = await litellm.acompletion(
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

    def completion(self, context, stream=False, temperature=0.5):
        return asyncio.run_coroutine_threadsafe(
            self.acompletion(context, stream, temperature),
            self._event_loop
        ).result()

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
        def truncate_content(content, max_image_length=100):
            if isinstance(content, str) and content.startswith("data:image"):
                return content[:max_image_length] + "..." if len(content) > max_image_length else content
            if isinstance(content, dict):
                return {k: truncate_content(v) for k, v in content.items()}
            if isinstance(content, list):
                return [truncate_content(item) for item in content]
            return content

        return "\n".join([f"{msg['role']}: {truncate_content(msg['content'])}" for msg in messages])


class CharacterLLMAgent(BaseLLMAgent):
    def __init__(self, 
        system_prompt, 
        model_name="gpt-4o-mini", 
        examples=None, 
        greetings=None, 
        control_agent=None, 
        event_loop=None
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt, 
            examples=examples,
            event_loop=event_loop
        )
        self.greetings = greetings
        self.control_agent = control_agent

    @staticmethod
    def from_config(config, **kwargs):
        required_fields = ["llm_model", "system_prompt"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        control_agent = None
        if "control_agent" in config:
            if config["control_agent"]["model"] == "pattern_matching":
                control_agent = ControlPatternAgent.from_config(config["control_agent"])
            else:
                control_agent = ControlLLMAgent.from_config(config["control_agent"])

        return CharacterLLMAgent(
            system_prompt=config["system_prompt"],
            model_name=config["llm_model"],
            examples=config.get("examples"),
            greetings=config.get("greetings"),
            control_agent=control_agent,
            **kwargs
        )

    def greeting_message(self):
        if not self.greetings:
            return None

        greeting = random.choice(self.greetings["choices"])
        # greeting is a dict like {"content": "...", "file": "..."} or a plain string
        if isinstance(greeting, dict):
            text = greeting["content"]
            file = greeting.get("file")
            if file and not os.path.exists(file):
                logger.warning("Greeting file {} does not exist", file)
                file = None
        else:
            text = greeting
            file = None

        return {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "voice_tone": self.greetings.get("voice_tone"),
            "file": file
        }

    async def acompletion(self, context, stream=False, temperature=0.5):
        assert self.control_agent is None or not stream, "Control agent does not support streaming"

        if self.control_agent is None:
            return await super().acompletion(context, stream, temperature)

        temperature_schedule = self.control_agent.temperature_schedule(temperature)
        for i_try, temperature in enumerate(temperature_schedule):
            logger.debug("Control agent try {}/{}, temperature {:.2f}", i_try + 1, len(temperature_schedule), temperature)
            response = await super().acompletion(context, temperature=temperature)
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

    def completion(self, context, stream=False, temperature=0.5):
        return asyncio.run_coroutine_threadsafe(
            self.acompletion(context, stream, temperature),
            self._event_loop
        ).result()


class CharacterEchoAgent:
    def __init__(self, *args, **kwargs):
        pass

    async def acompletion(self, context, stream=False, temperature=0.5):
        return context.get_messages()[-1]["content"][0]["text"]

    def completion(self, context, stream=False, temperature=0.5):
        return context.get_messages()[-1]["content"][0]["text"]

    def greeting_message(self):
        return {
            "role": "assistant", 
            "content": [{"type": "text", "text": "Hello, I'm an echo agent"}]
        }


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
        context.add_message({"role": "user", "content": [{"type": "text", "text": response_text}]})
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