from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from sys import stderr
from dataclasses import dataclass

from openrlhf.utils.interface import AsyncLLMInterface, AgentConversation, Message


COST_PER_INPUT_TOKEN = {
    "openai": {
        "gpt-4.1": 2.0 / 1e6,
        "gpt-4.1-mini": 0.4 / 1e6,
        "gpt-4.1-nano": 0.1 / 1e6,
        "gpt-4o": 2.5 / 1e6,
        "gpt-4o-mini": 0.15 / 1e6,
        "o3": 2.0 / 1e6,
        "o4-mini": 1.1 / 1e6,
    },
    "anthropic": {
        "sonnet3.7": 3.0 / 1e6,
        "sonnet3.5": 3.0 / 1e6,
        "haiku3.5": 0.8 / 1e6,
        "sonnet4": 3.0 / 1e6,
        "haiku3": 0.25 / 1e6,
        "opus4": 15.0 / 1e6,
        "opus3": 15.0 / 1e6,
    },
}


COST_PER_CACHED_INPUT_TOKEN = {
    "openai": {
        "gpt-4.1": 0.5 / 1e6,
        "gpt-4.1-mini": 0.1 / 1e6,
        "gpt-4.1-nano": 0.025 / 1e6,
        "gpt-4o": 1.25 / 1e6,
        "gpt-4o-mini": 0.075 / 1e6,
        "o3": 0.5 / 1e6,
        "o4-mini": 0.275 / 1e6,
    },
}


COST_PER_OUTPUT_TOKEN = {
    "openai": {
        "gpt-4.1": 8.0 / 1e6,
        "gpt-4.1-mini": 1.6 / 1e6,
        "gpt-4.1-nano": 0.4 / 1e6,
        "gpt-4o": 10.0 / 1e6,
        "gpt-4o-mini": 0.6 / 1e6,
        "o3": 8.0 / 1e6,
        "o4-mini": 4.4 / 1e6,
    },
    "anthropic": {
        "sonnet3.7": 15.0 / 1e6,
        "sonnet3.5": 15.0 / 1e6,
        "haiku3.5": 4.0 / 1e6,
        "sonnet4": 15.0 / 1e6,
        "haiku3": 1.25 / 1e6,
        "opus4": 75.0 / 1e6,
        "opus3": 75.0 / 1e6,
    },
}


ANTHROPIC_MODEL_NAMES = {
    "claude-opus-4-20250514": "opus4",
    "claude-opus-4-0": "opus4",
    "claude-sonnet-4-20250514": "sonnet4",
    "claude-sonnet-4-0": "sonnet4",
    "claude-sonnet-3.7-20250219": "sonnet3.7",
    "claude-3-7-sonnet-latest": "sonnet3.7",
    "claude-3.5-haiku-20241022": "haiku3.5",
    "claude-3-5-haiku-latest": "haiku3.5",
    "claude-3-haiku-20240307": "haiku3",
    "claude-3-5-sonnet-latest": "sonnet3.5",
    "claude-3-5-sonnet-20240620": "sonnet3.5",
    "claude-3-5-sonnet-20241022": "sonnet3.5",
}


def openai_or_anthropic_api_cost(
    model_provider: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> float | int:
    if model_provider == "openai":
        return (
            input_tokens * COST_PER_INPUT_TOKEN[model_provider][model_name]
            + output_tokens * COST_PER_OUTPUT_TOKEN[model_provider][model_name]
            + cached_input_tokens * COST_PER_CACHED_INPUT_TOKEN[model_provider][model_name]
        )
    elif model_provider == "anthropic":
        return (
            input_tokens * COST_PER_INPUT_TOKEN[model_provider][model_name]
            + output_tokens * COST_PER_OUTPUT_TOKEN[model_provider][model_name]
        )
    else:
        raise ValueError(f"Unknown model provider: {model_provider}")


total_cost = 0.0
total_cost_lock = asyncio.Lock()


@dataclass(frozen=True)
class AsyncOpenAIOrAnthropicLLM(AsyncLLMInterface):
    client: AsyncOpenAI | AsyncAnthropic
    model: str
    temperature: float
    max_completion_tokens: int

    async def generate_assistant_message(
        self,
        conversation: AgentConversation,
        stop_strings: list[str] | None,
    ) -> None:
        messages = self._merge_tool_and_user_messages(conversation.messages)

        completion_text: str = await self._generate_completion(messages, stop_strings=stop_strings)

        # the together.ai api ignores stop strings
        # (note: i don't know if it always does or sometimes does)
        together_ai_api = self.client.base_url is not None and "api.together.xyz" in str(self.client.base_url)
        if together_ai_api and stop_strings is not None:
            for stop_string in stop_strings:
                if stop_string in completion_text:
                    completion_text = completion_text[: completion_text.index(stop_string)]

        conversation.messages.append({"role": "assistant", "content": completion_text})

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=15, min=1),
        before_sleep=lambda retry_state: print(
            f"Calling OpenAI or Anthropic API: Attempt {retry_state.attempt_number} Failed: Exception: {retry_state.outcome.exception()}",
            file=stderr,
        ),
    )
    async def _generate_completion(self, messages: list[Message], stop_strings: list[str] | None) -> str:

        global total_cost_lock, total_cost

        if isinstance(self.client, AsyncOpenAI):
            completion = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                stop=stop_strings if self.model != "o3" else None,
            )

            print(completion.usage)

            cost = openai_or_anthropic_api_cost(
                model_provider="openai",
                model_name=self.model,
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
                cached_input_tokens=0,
            )

            async with total_cost_lock:
                total_cost += cost
                print(f"total api call cost: ${total_cost}")

            return completion.choices[0].message.content

        if isinstance(self.client, AsyncAnthropic):
            completion = await self.client.messages.create(
                messages=messages[1:] if messages[0]["role"] == "system" else messages,
                system=[{"type": "text", "text": messages[0]["content"], "cache_control": {"type": "ephemeral"}}]
                if messages[0]["role"] == "system"
                else None,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens,
                stop_sequences=stop_strings,
            )

            cost = openai_or_anthropic_api_cost(
                model_provider="anthropic",
                model_name=self.model,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
                cached_input_tokens=0,
            )

            async with total_cost_lock:
                total_cost += cost
                print(f"total api call cost: ${total_cost}")

            return completion.content[0].text

        raise TypeError(
            f"AsyncOpenAIOrAnthropicLLM.client should be of type OpenAI or Anthropic, but found type {type(self.client)}."
        )

    def _merge_tool_and_user_messages(self, messages: list[Message]) -> list[Message]:
        merged_messages = []

        for message in messages:
            if message["role"] == "tool":
                assert set(message.keys()) == {"role", "content"}
                message = {"role": "user", "content": f"<tool_call>\n{message['content']}\n<tool_call/>"}

            if len(merged_messages) > 0 and message["role"] == merged_messages[-1]["role"]:
                assert set(message.keys()) == {"role", "content"}
                merged_messages[-1]["content"] += "\n\n" + message["content"]
                continue

            merged_messages.append(message)

        return merged_messages
