from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import count
import threading
from time import perf_counter
from typing import *
import ray.remote_function
import vllm
from vllm import CompletionOutput, SamplingParams
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    apply_hf_chat_template,
    parse_chat_messages,
    resolve_chat_template_content_format,
)
from vllm.inputs import TokensPrompt
from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import ray
import logging
from dataclasses import dataclass
from sys import stderr
from pymongo import MongoClient
from datetime import datetime
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

Message = Dict[str, str]
Reward = float
AgentState = Any  # State needed to track conversation progress


@dataclass
class DelayedFunction:
    function: Callable
    remote_function: ray.remote_function.RemoteFunction
    args: Iterable[Any]
    kwargs: dict[str, Any]


@dataclass
class AgentConversation:
    messages: List[Message]
    tokens_by_turn: List[Dict[str, Any]]
    first_prompt_tokens: List[int]
    all_tokens: List[int]
    total_tokens: int

    @staticmethod
    def empty() -> "AgentConversation":
        return AgentConversation(messages=[], tokens_by_turn=[], first_prompt_tokens=[], all_tokens=[], total_tokens=0)

    def append_assistant_message_and_tokens(self, vllm_output: RequestOutput) -> None:
        self.messages.append({"role": "assistant", "content": vllm_output.outputs[0].text})
        output_tokens = vllm_output.outputs[0].token_ids
        input_tokens = vllm_output.prompt_token_ids[self.total_tokens :]
        self.tokens_by_turn.append({"input_tokens": input_tokens, "output_tokens": output_tokens})
        if self.total_tokens == 0:
            self.first_prompt_tokens = vllm_output.prompt_token_ids
        self.all_tokens = list(vllm_output.prompt_token_ids) + list(vllm_output.outputs[0].token_ids)
        self.total_tokens += len(input_tokens) + len(output_tokens)


class AgentInterface(ABC):
    def __init__(
        self,
        full_data: list[dict],
        sampling_params: SamplingParams,
        llm_engine: vllm.AsyncLLMEngine | OpenAI | Anthropic,
        async_event_loop: asyncio.AbstractEventLoop,
        # truncate_prompt_tokens: int | None = None, # just let the user provide this in sampling_params. uncomment this (and the subsequent commented block in this function which handles those two parameters) if commenting this created a bug
        # stop_on_truncation: bool = False,
        stop_strings: list[str] | None = None,
        length_penalty: float = 0.0,
        max_steps: int | None = None,
    ) -> None:
        self.full_data = full_data
        self.sampling_params = sampling_params
        self.llm_engine = llm_engine
        self.async_event_loop = async_event_loop
        # if isinstance(llm_engine, vllm.LLM):
        #     self.thread_safe_llm = ThreadSafeLLM(llm_engine)
        self.length_penalty = length_penalty
        self.stop_strings = stop_strings
        self.max_steps = max_steps

        if stop_strings is not None:
            self.sampling_params.stop = stop_strings
            self.sampling_params.include_stop_str_in_output = True

        self._llm_engine_request_id_counter = 0

        # just let the user provide this in sampling_params. uncomment this (and the arguments truncate_prompt_tokens and stop_on_truncation to __init__) if commenting this created a bug
        # self.truncate_prompt_tokens = truncate_prompt_tokens
        # self.stop_on_truncation = stop_on_truncation
        # if truncate_prompt_tokens is not None:
        #    if isinstance(self.llm_engine, vllm.LLM):
        #         self.sampling_params.truncate_prompt_tokens = truncate_prompt_tokens
        #         logger.info(f"Set SamplingParams.truncate_prompt_tokens to {truncate_prompt_tokens}")
        #    else:
        #         logger.warning(
        #             f"truncate_prompt_tokens ({truncate_prompt_tokens}) provided but LLM engine is not vLLM. "
        #             "This parameter will be ignored for both SamplingParams and manual truncation."
        #         )

    @abstractmethod
    def init_state(self, data: dict) -> AgentState:
        """Initialize the state for a new RL env, given a dict of the info in one element of the dataset"""
        pass

    @abstractmethod
    def get_next_prompt(
        self, messages: list[Message], state: AgentState
    ) -> tuple[list[Message] | Message | None, AgentState]:
        """Input:
        - messages: the messages in the conversation
        - state: the state of the environment

        Output:
        - next_prompt: the next prompt to send to the model (can be a list of prompts)
        - next_state: the updated state of the environment

        Note: an output of None means that the environment is done and the agent should stop generating.

        Get the next prompt to send to the model and updated state.
        In this function, you should (1) use the model's last message to update the state.
        Then (2) create the prompt to send to the model, which should probably incorporate observations about the environment.
        Finally, (3) return the next prompt for the model to send, along with the updated state."""
        pass

    @abstractmethod
    def is_done(self, messages: List[Message], state: AgentState) -> bool:
        """Determine if the conversation is complete"""
        pass

    @abstractmethod
    def get_reward(self, messages: List[Message], state: AgentState) -> Reward:
        """Get the reward for the conversation.
        NOTE: This should not include length penalty!"""
        pass

    async def generate_many(self) -> list[tuple[AgentConversation, Reward]]:
        self.generation_id = 0
        self.start_time = perf_counter()

        # async def workload():
        #     return await asyncio.gather(*[self._generate_single_rollout(data) for data in self.full_data])

        # return asyncio.run(workload())

        # with ThreadPoolExecutor() as executor:
        #     return list(executor.map(self._generate_single_rollout, self.full_data))

        return [
            self.async_event_loop.run_until_complete(self._generate_single_rollout(data)) for data in self.full_data
        ]

    async def _generate_single_rollout(self, data: dict) -> tuple[AgentConversation, Reward]:
        state = self.init_state(data)
        conversation = AgentConversation.empty()

        for step in range(self.max_steps) if self.max_steps is not None else count():
            x = self.get_next_prompt(messages=conversation.messages, state=state)

            if x is None:
                break
            new_user_messages, state = x
            if new_user_messages is None:
                break
            if not isinstance(new_user_messages, list):
                new_user_messages = [new_user_messages]

            conversation.messages += new_user_messages

            generation_id = self.generation_id
            self.generation_id += 1
            print(f"STARTING generation: state:{state} id:{generation_id} time:{perf_counter() - self.start_time}")
            vllm_output = await self._generate_chat_completion(conversation.messages)
            conversation.append_assistant_message_and_tokens(vllm_output)
            print(f"FINISHED generation: state:{state} id:{generation_id} time:{perf_counter() - self.start_time}")

            if self.is_done(messages=conversation.messages, state=state):
                break

        reward = self.get_reward(messages=conversation.messages, state=state)

        return conversation, reward

    async def _chat_with_truncation(
        self,
        messages: list[Message],
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        truncation_amt: Optional[int] = None,
    ) -> Tuple[RequestOutput, bool] | RequestOutput:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the :meth:`generate` method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A list of conversations or a single conversation.

              - Each conversation is represented as a list of messages.
              - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
              If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

              - "string" will render the content as a string.
                Example: ``"Who are you?"``
              - "openai" will render the content as a list of dictionaries,
                similar to OpenAI schema.
                Example: ``[{"type": "text", "text": "Who are you?"}]``

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                ``True`` if ``add_generation_prompt`` is also ``True``.
            mm_processor_kwargs: Multimodal processor kwarg overrides for this
                chat request. Only used for offline requests.

        Returns:
            A list of ``RequestOutput`` objects containing the generated
            responses in the same order as the input messages.
            Optionally, a list of booleans indicating whether each prompt was truncated.
        """

        print("DEBUG: 1")

        tokenizer = await self.llm_engine.get_tokenizer()
        model_config = await self.llm_engine.get_model_config()
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tools,
            chat_template_content_format,
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )

        print("DEBUG: 2")

        # NOTE: _parse_chat_message_content_parts() currently doesn't
        # handle mm_processor_kwargs, since there is no implementation in
        # the chat message parsing for it.
        conversation, _ = parse_chat_messages(
            messages,
            model_config,
            tokenizer,
            content_format=resolved_content_format,
        )

        print("DEBUG: 3")

        prompt_str = apply_hf_chat_template(
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            conversation=conversation,
            chat_template=chat_template,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        # Special tokens are already included in chat templates so
        # should not be added by the tokenizer in this case.
        prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

        print("DEBUG: 4")

        # Truncate the prompt if necessary
        truncate = truncation_amt is not None and len(prompt_token_ids) > truncation_amt
        if truncate:
            old_len = len(prompt_token_ids)
            prompt_token_ids = prompt_token_ids[:truncation_amt]
            logger.warning(f"Truncated prompt from {old_len} tokens to {truncation_amt} tokens.")

        print("DEBUG: 5")

        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        print("DEBUG: 6")

        # TO DO: with self._lock():
        request_id = str(self._llm_engine_request_id_counter)
        self._llm_engine_request_id_counter += 1

        print("DEBUG: 7")

        outputs = self.llm_engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )

        print("DEBUG: 8")

        print(f"{prompt=} {sampling_params=} {request_id=} {lora_request=}")
        print(f"{outputs=}")

        async for output in outputs:
            print(f"{output.request_id} {output.outputs[0].text}")
            final_outputs = output

        if truncation_amt is not None:
            return final_outputs, truncate
        else:
            return final_outputs

    async def _generate_chat_completion(self, messages: list[Message]) -> RequestOutput:
        return await self._chat_with_truncation(
            messages=messages,
            sampling_params=self.sampling_params,
            truncation_amt=self.sampling_params.truncate_prompt_tokens,
        )
        # return self.thread_safe_llm.chat_with_truncation(
        #     messages=messages,
        #     sampling_params=self.sampling_params,
        #     truncation_amt=self.sampling_params.truncate_prompt_tokens,
        # )


class ThreadSafeLLM:
    def __init__(self, llm_engine: vllm.AsyncLLMEngine) -> None:
        self.llm_engine = llm_engine
        self._lock = threading.Lock()
        self.request_id_counter = 0

    def chat_with_truncation(
        self,
        messages: list[Message],
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        truncation_amt: Optional[int] = None,
    ) -> Tuple[RequestOutput, bool] | RequestOutput:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the :meth:`generate` method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A list of conversations or a single conversation.

              - Each conversation is represented as a list of messages.
              - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
              If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

              - "string" will render the content as a string.
                Example: ``"Who are you?"``
              - "openai" will render the content as a list of dictionaries,
                similar to OpenAI schema.
                Example: ``[{"type": "text", "text": "Who are you?"}]``

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                ``True`` if ``add_generation_prompt`` is also ``True``.
            mm_processor_kwargs: Multimodal processor kwarg overrides for this
                chat request. Only used for offline requests.

        Returns:
            A list of ``RequestOutput`` objects containing the generated
            responses in the same order as the input messages.
            Optionally, a list of booleans indicating whether each prompt was truncated.
        """
        tokenizer = self.llm_engine.get_tokenizer()
        model_config = self.llm_engine.llm_engine.get_model_config()
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tools,
            chat_template_content_format,
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )

        # NOTE: _parse_chat_message_content_parts() currently doesn't
        # handle mm_processor_kwargs, since there is no implementation in
        # the chat message parsing for it.
        conversation, _ = parse_chat_messages(
            messages,
            model_config,
            tokenizer,
            content_format=resolved_content_format,
        )

        prompt_str = apply_hf_chat_template(
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            conversation=conversation,
            chat_template=chat_template,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        # Special tokens are already included in chat templates so
        # should not be added by the tokenizer in this case.
        prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

        # Truncate the prompt if necessary
        truncate = truncation_amt is not None and len(prompt_token_ids) > truncation_amt
        if truncate:
            old_len = len(prompt_token_ids)
            prompt_token_ids = prompt_token_ids[:truncation_amt]
            logger.warning(f"Truncated prompt from {old_len} tokens to {truncation_amt} tokens.")

        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        outputs = self.llm_engine.generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0]

        if truncation_amt is not None:
            return outputs, truncate
        else:
            return outputs

    def generate(
        self, prompt: str, sampling_params: SamplingParams, lora_request: LoRARequest | None = None
    ) -> list[RequestOutput]:
        with self._lock:
            self.llm_engine.llm_engine.add_request(
                self.request_id_counter, prompt, sampling_params, lora_request=lora_request
            )
            self.request_id_counter += 1


class TimeMetrics:
    time_initializing_environments: Optional[float]
    time_doing_environment_steps: Optional[List[float]]
    time_generating_completions: Optional[List[float]]
    time_evaluating_is_done: Optional[List[float]]
    time_computing_rewards: Optional[float]
    everything_start_time: Optional[float]
    init_env_start_time: Optional[float]
    compute_rewards_start_time: Optional[float]
    total_time: Optional[float]

    def __init__(self):
        self.everything_start_time = perf_counter()

        self.time_doing_environment_steps = []
        self.time_generating_completions = []
        self.time_evaluating_is_done = []

    def finish_and_log(self):
        self.total_time = perf_counter() - self.everything_start_time

        logger.info(f"Rollout completed in {int(self.total_time)} seconds. Breakdown of time spent:")
        logger.info(
            f"Generating completions with vllm: {int(sum(self.time_generating_completions))} seconds ({sum(self.time_generating_completions) / self.total_time:.0%}, breakdown by step: {', '.join(str(int(t)) for t in self.time_generating_completions)})"
        )
        logger.info(
            f"Initializing environments: {int(self.time_initializing_environments)} seconds ({self.time_initializing_environments / self.total_time:.0%})"
        )
        logger.info(
            f"Doing environment steps: {int(sum(self.time_doing_environment_steps))} seconds ({sum(self.time_doing_environment_steps) / self.total_time:.0%}, breakdown by step: {', '.join(str(int(t)) for t in self.time_doing_environment_steps)})"
        )
        logger.info(
            f"Evaluating whether environments are done: {int(sum(self.time_evaluating_is_done))} seconds ({sum(self.time_evaluating_is_done) / self.total_time:.0%}, breakdown by step: {', '.join(str(int(t)) for t in self.time_evaluating_is_done)})"
        )
        logger.info(
            f"Computing rewards: {int(self.time_computing_rewards)} ({self.time_computing_rewards / self.total_time:.0%})"
        )
        unaccounted_for_time = (
            self.total_time
            - sum(self.time_generating_completions)
            - self.time_initializing_environments
            - sum(self.time_doing_environment_steps)
            - sum(self.time_evaluating_is_done)
            - self.time_computing_rewards
        )
        logger.info(
            f"Unaccounted for (should be close to zero): {int(unaccounted_for_time)} ({unaccounted_for_time / self.total_time:.0%})"
        )
