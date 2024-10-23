import threading
import json
from datetime import datetime
import litellm
from loguru import logger

from prompts import response_agent_system_prompt, response_agent_examples, response_agent_greeting_message


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
                return True

            if text_compare_f is None:
                text_compare_f = lambda x, y: x == y

            if not text_compare_f(self.messages[-1]["content"], message["content"]):
                self.messages[-1]["content"] = message["content"]
                self.messages[-1]["handled"] = False
                return True

            return False

    def get_messages(self, include_fields=None, filter=None):
        if include_fields is None:
            messages = self.messages
        else:
            messages = [{k: v for k, v in msg.items() if k in include_fields} for msg in self.messages]

        if filter:
            messages = [msg for msg in messages if filter(msg)]

        return messages

    def update_message(self, id, **kwargs):
        for msg in self.messages:
            if msg["id"] == id:
                msg.update(kwargs)
                return True
        return False

    def to_text(self):
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

    def to_json(self, filter=None):
        messages = self.get_messages(filter=filter)
        return json.loads(json.dumps(messages, default=self._json_serializer))

    @staticmethod
    def _json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.strftime("%H:%M:%S")
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class BaseLLMAgent:
    def __init__(self, model_name, system_prompt, examples=None, output_json=False):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.examples = examples
        self.output_json = output_json

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

        logger.error("Messages:\n{}", self._messages_to_text(messages))
        
        response = litellm.completion(
            model=self.model_name, 
            messages=messages, 
            response_format={"type": "json_object"} if self.output_json else None,
            temperature=0.5
        )

        r_content = response.choices[0].message.content
        logger.debug("Response content: {}", r_content)

        if self.output_json:
            response = json.loads(r_content)
        else:
            response = r_content

        return {
            "response": response,
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
    default_response = ""
    
    def __init__(self):
        super().__init__(
            model_name="openai/openai-gpt-4o-mini", 
            system_prompt=response_agent_system_prompt, 
            examples=response_agent_examples,
            output_json=False
        )

    # def _extra_context_to_text(self, context):
    #     assert "detection_agent_state" in context, "This agent requires detection agent's current state"
    #     assert "clarification_agent_state" in context, "This agent requires clarification agent's current state"
    #     d_state = context["detection_agent_state"]
    #     c_state = context["clarification_agent_state"]
    #     text = "Detection agent state:\nAction: {}\nReason: {}\n\n".format(d_state["action"], d_state["reason"])
    #     text += "Clarification agent state:\nHas question: {}\nQuestion: {}\n\n".format(c_state["has_question"], c_state["question"])
    #     return text

    def completion(self, context, *args, **kwargs):
        result = super().completion(context, *args, **kwargs)

        if result["response"] is None:
            result["response"] = self.default_response

        return result

    @staticmethod
    def greeting_message():
        return {
            "role": "assistant",
            "content": response_agent_greeting_message,
            "time": datetime.now()
        }
