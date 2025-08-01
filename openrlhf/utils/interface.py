from abc import ABC, abstractmethod
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import count, pairwise, zip_longest
from time import perf_counter
from typing import *
from uuid import uuid4
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
from plotly.graph_objects import Figure
import traceback
import json
import ray
import logging
from dataclasses import dataclass, field
from datetime import datetime
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

Message = Dict[str, str]
Reward = float
AgentState = Any  # State needed to track conversation progress


@dataclass
class AgentConversation:
    env_name: str
    messages: list[Message] = field(default_factory=lambda: [])
    tokens_by_turn: list[dict[str, Any]] = field(default_factory=lambda: [])  # to do: better type hint than Any
    first_prompt_tokens: list[int] = field(default_factory=lambda: [])
    all_tokens: list[int] = field(default_factory=lambda: [])
    n_tokens: int = 0
    n_assistant_tokens: int = 0
    was_truncated: bool = False
    extra_metrics: dict[str, float] | None = field(default_factory=lambda: {"n_errors": 0.0, "num_steps": 0.0})
    error: bool = False
    action_mask: list[int] = field(default_factory=lambda: [0])
    num_actions_list: list[int] = field(default_factory=lambda: [])

    def increment_num_steps(self) -> None:
        if self.extra_metrics is None:
            self.extra_metrics = {}
        self.extra_metrics["num_steps"] = self.extra_metrics.get("num_steps", 0.0) + 1.0


class AsyncLLMInterface(ABC):
    @abstractmethod
    async def generate_assistant_message(
        self, conversation: AgentConversation, stop_strings: list[str] | None
    ) -> None:
        pass


@dataclass(frozen=True)
class AsyncVLLM(AsyncLLMInterface):
    llm_engine: vllm.AsyncLLMEngine
    sampling_params: SamplingParams

    async def generate_assistant_message(
        self,
        conversation: AgentConversation,
        stop_strings: list[str] | None,
    ) -> None:
        sampling_params = self.sampling_params
        if stop_strings is not None:
            sampling_params = deepcopy(self.sampling_params)
            sampling_params.stop = stop_strings
            sampling_params.include_stop_str_in_output = True

        output, truncated_tokens = await _vllm_chat_with_truncation(
            llm_engine=self.llm_engine, messages=conversation.messages, sampling_params=sampling_params
        )
        was_truncated = truncated_tokens > 0

        if conversation.n_tokens == 0:
            conversation.first_prompt_tokens = output.prompt_token_ids

        input_tokens = output.prompt_token_ids[conversation.n_tokens :]
        output_tokens = output.outputs[0].token_ids

        output_message = {"role": "assistant", "content": output.outputs[0].text}
        conversation.messages.append(output_message)
        conversation.tokens_by_turn.append({"input_tokens": input_tokens, "output_tokens": output_tokens})
        conversation.n_tokens += len(input_tokens) + len(output_tokens)
        conversation.n_assistant_tokens += len(output_tokens)

        conversation.action_mask.extend([0] * len(input_tokens))
        if was_truncated:
            conversation.action_mask = conversation.action_mask[: sampling_params.truncate_prompt_tokens] + [0]
        conversation.action_mask.extend([1] * len(output_tokens))

        conversation.num_actions_list.append(len(output_tokens))

        conversation.all_tokens = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)

        if was_truncated:
            conversation.was_truncated = True


async def _vllm_chat_with_truncation(
    llm_engine: vllm.AsyncLLMEngine,
    messages: list[Message],
    sampling_params: SamplingParams,
    lora_request: Optional[LoRARequest] = None,
    chat_template: Optional[str] = None,
    chat_template_content_format: ChatTemplateContentFormatOption = "auto",
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[RequestOutput, bool]:
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
        Also, the number of truncated tokens (Can be zero)
    """

    tokenizer = await llm_engine.get_tokenizer()
    model_config = await llm_engine.get_model_config()
    resolved_content_format = resolve_chat_template_content_format(
        chat_template,
        tools,
        chat_template_content_format,
        tokenizer,
        model_config=model_config,
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
        model_config=model_config,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )
    # Special tokens are already included in chat templates so
    # should not be added by the tokenizer in this case.
    prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

    # Truncate the prompt if necessary
    was_truncated = (
        sampling_params.truncate_prompt_tokens is not None
        and len(prompt_token_ids) > sampling_params.truncate_prompt_tokens
    )
    if was_truncated:
        old_len = len(prompt_token_ids)
        prompt_token_ids = prompt_token_ids[: sampling_params.truncate_prompt_tokens]
        num_truncated_tokens = old_len - sampling_params.truncate_prompt_tokens
        logger.warning(f"Truncated prompt from {old_len} tokens to {sampling_params.truncate_prompt_tokens} tokens.")
    else:
        num_truncated_tokens = 0

    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

    request_id = str(uuid4())

    output_generator = llm_engine.generate(
        prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        lora_request=lora_request,
    )

    finished_output = None

    async for output in output_generator:
        assert output.request_id == request_id
        if not output.finished:
            continue
        assert finished_output is None
        finished_output = output

    assert finished_output is not None

    return finished_output, num_truncated_tokens


class AgentInterface(ABC):
    def __init__(
        self,
        stop_strings: list[str] | None = None,
        max_steps: int | None = None,
        stop_on_truncation: bool = False,
        save_rollout_time_statistics_directory: str | None = "/root/rollout-time-statistics/",
        vllm_engine_index: int = 0,
        compact_filtering: bool = False,
        filter_max_steps: bool = False,
    ) -> None:
        self.stop_strings = stop_strings
        self.max_steps = max_steps
        self.stop_on_truncation = stop_on_truncation
        self.save_rollout_time_statistics_directory = save_rollout_time_statistics_directory
        self.num_errors = 0
        self.errors = []
        self.vllm_engine_index = vllm_engine_index
        self.compact_filtering = compact_filtering
        self.filter_max_steps = filter_max_steps

    @abstractmethod
    async def init_all_states(self, full_data: list[dict]) -> list[AgentState]:
        """Initialize the states for a new RL environments, given a list of dict elements of the dataset"""
        pass

    @abstractmethod
    async def cleanup_all_states(self, all_states: list[AgentState], full_data: list[dict]) -> None:
        pass

    @abstractmethod
    async def get_next_prompt(
        self, messages: List[Message], state: AgentState, remaining_steps: int | None
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
    async def is_done(self, messages: List[Message], state: AgentState) -> bool:
        """Determine if the conversation is complete"""
        pass

    @abstractmethod
    async def get_reward(self, messages: List[Message], state: AgentState) -> Reward | None:
        """Get the reward for the conversation.
        NOTE: This should not include length penalty!"""
        pass

    async def get_extra_metrics(self, messages: list[Message], state: AgentState) -> dict[str, float]:
        return {}

    async def get_reward_in_eval(self, messages: List[Message], state: AgentState) -> Reward | None:
        """Get the eval reward for the conversation. Used if the train reward may reflect a different thing than what we'd like to measure"""
        return await self.get_reward(messages, state)

    async def generate_rollouts(
        self, llm: AsyncLLMInterface, full_data: list[dict], env_name: str, is_eval: bool = False
    ) -> list[tuple[AgentConversation, Reward | None]]:
        time_init_env_started = perf_counter()
        try:
            states = await self.init_all_states(full_data)
        except Exception as e:
            self.num_errors += 1
            self.errors.append(f"Error in init_all_states: {str(e)}")
            logger.error(f"Error in init_all_states: {str(e)}")
            return [
                (
                    AgentConversation(
                        env_name=env_name, extra_metrics={"n_errors": 1.0, "num_steps": 0.0}, error=True
                    ),
                    None,
                )
                for _ in range(len(full_data))
            ]

        results = await asyncio.gather(
            *[
                self._generate_single_rollout(
                    llm=llm,
                    initial_state=state,
                    time_init_env_started=time_init_env_started,
                    env_name=env_name,
                    is_eval=is_eval,
                )
                for state in states
            ]
        )

        if self.save_rollout_time_statistics_directory is not None:
            try:
                os.makedirs(self.save_rollout_time_statistics_directory, exist_ok=True)
                make_rollout_time_statistics_plot(
                    stats=[stats for _, _, stats, _ in results],
                    save_filename=os.path.join(
                        self.save_rollout_time_statistics_directory,
                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".html",
                    ),
                )
            except Exception as e:
                print("An exception occurred when trying to plot the rollout time statistics.")
                print("The exception is:", e)
                traceback.print_exc()

        try:
            await self.cleanup_all_states(all_states=[state for _, _, _, state in results], full_data=full_data)
        except Exception as e:
            self.num_errors += 1
            self.errors.append(f"Error in cleanup_all_states: {str(e)}")
            logger.error(f"Error in cleanup_all_states: {str(e)}")
            for conversation, reward, stats, state in results:
                conversation.error = True

        for conversation, reward, stats, state in results:
            conversation.extra_metrics["n_errors"] = float(conversation.error)

        return [(conversation, reward) for conversation, reward, stats, state in results]

    async def _generate_single_rollout(
        self,
        llm: AsyncLLMInterface,
        initial_state: AgentState,
        time_init_env_started: float,
        env_name: str,
        is_eval: bool = False,
    ) -> tuple[AgentConversation, Reward | None, "RolloutTimeStatistics", AgentState | None]:
        stats = RolloutTimeStatistics(time_init_env_started=time_init_env_started)

        conversation = AgentConversation(env_name=env_name)
        state = initial_state

        was_truncated = False
        hit_max_steps = False

        for step in count():
            stats.on_env_step_start()
            try:
                new_messages, state = await self.get_next_prompt(
                    messages=conversation.messages,
                    state=state,
                    remaining_steps=self.max_steps - step if self.max_steps is not None else None,
                )
            except Exception as e:
                self.num_errors += 1
                self.errors.append(f"Error in get_next_prompt: {str(e)}")
                logger.error(f"Error in get_next_prompt: {str(e)} {traceback.format_exc()}")
                conversation.error = True
                break

            if new_messages is None:
                break
            if not isinstance(new_messages, list):
                new_messages = [new_messages]

            if self.max_steps is not None and step >= self.max_steps:
                hit_max_steps = True
                break
            stats.on_computing_is_done_start()
            try:
                is_done = await self.is_done(messages=conversation.messages, state=state)
            except Exception as e:
                self.num_errors += 1
                self.errors.append(f"Error in is_done: {str(e)}")
                logger.error(f"Error in is_done: {str(e)}")
                conversation.error = True
                break

            if is_done:
                break
            if self.stop_on_truncation and conversation.was_truncated:
                was_truncated = True
                break

            conversation.messages += new_messages
            conversation.increment_num_steps()

            stats.on_llm_completion_start()
            await llm.generate_assistant_message(conversation, stop_strings=self.stop_strings)

        stats.on_computing_reward_start()

        # Normal reward calculation
        try:
            if is_eval:
                reward = await self.get_reward_in_eval(messages=conversation.messages, state=state)
            else:
                reward = await self.get_reward(messages=conversation.messages, state=state)
        except Exception as e:
            self.num_errors += 1
            self.errors.append(f"Error in get_reward: {str(e)}")
            logger.error(f"Error in get_reward: {str(e)}")
            conversation.error = True
            reward = None

        if was_truncated and self.compact_filtering:
            reward = None
        elif hit_max_steps and self.filter_max_steps:
            reward = None

        stats.on_finish()

        try:
            extra_metrics = await self.get_extra_metrics(messages=conversation.messages, state=state)
            assert isinstance(extra_metrics, dict)
            assert all(isinstance(key, str) for key in extra_metrics.keys())
            assert all(isinstance(value, float) for value in extra_metrics.values())
            conversation.extra_metrics = conversation.extra_metrics | extra_metrics
        except Exception as e:
            self.num_errors += 1
            self.errors.append(f"Error in get_extra_metrics {str(e)}")
            logger.error(f"Error in get_extra_metrics {str(e)}")
            conversation.error = True

        conversation.action_mask = conversation.action_mask[1:]
        return conversation, reward, stats, state


@dataclass(frozen=True)
class TimeInterval:
    start: float
    end: float
    description: str


@dataclass
class RolloutTimeStatistics:
    time_init_env_started: float | None = None
    times_env_steps_started: list[float] = field(default_factory=lambda: [])
    times_computing_is_done_started: list[float] = field(default_factory=lambda: [])
    times_llm_completions_started: list[float] = field(default_factory=lambda: [])
    time_computing_reward_started: float | None = None
    time_finished: float | None = None

    def on_init_env_start(self) -> None:
        self.time_init_env_started = perf_counter()

    def on_env_step_start(self) -> None:
        self.times_env_steps_started.append(perf_counter())

    def on_computing_is_done_start(self) -> None:
        self.times_computing_is_done_started.append(perf_counter())

    def on_llm_completion_start(self) -> None:
        self.times_llm_completions_started.append(perf_counter())

    def on_computing_reward_start(self) -> None:
        self.time_computing_reward_started = perf_counter()

    def on_finish(self) -> None:
        self.time_finished = perf_counter()

    def time_intervals(self) -> list[TimeInterval]:
        assert self.time_init_env_started is not None
        assert self.time_computing_reward_started is not None
        assert self.time_finished is not None
        assert len(self.times_env_steps_started) > 0
        assert (
            len(self.times_env_steps_started)
            >= len(self.times_computing_is_done_started)
            >= len(self.times_llm_completions_started)
        )
        assert len(self.times_env_steps_started) <= len(self.times_llm_completions_started) + 1

        times: list[float] = []
        times.append(self.time_init_env_started)
        for step_times in zip_longest(
            self.times_env_steps_started,
            self.times_computing_is_done_started,
            self.times_llm_completions_started,
            fillvalue=None,
        ):
            for time in step_times:
                if time is None:
                    continue
                times.append(time)
        times.append(self.time_computing_reward_started)
        times.append(self.time_finished)

        assert (
            len(times)
            == len(self.times_env_steps_started)
            + len(self.times_computing_is_done_started)
            + len(self.times_llm_completions_started)
            + 3
        )

        descriptions = (
            ["initializing environment"]
            + (
                ["environment step", "computing is done", "generating llm completion"]
                * len(self.times_env_steps_started)
            )[: len(times) - 3]
            + ["computing reward"]
        )

        return [
            TimeInterval(start=start_time, end=end_time, description=description)
            for (start_time, end_time), description in zip(pairwise(times), descriptions, strict=True)
        ]


def make_rollout_time_statistics_plot(stats: list[RolloutTimeStatistics], save_filename: str) -> None:
    description_to_color = {
        "initializing environment": "lightblue",
        "environment step": "blue",
        "computing is done": "cyan",
        "generating llm completion": "red",
        "computing reward": "darkblue",
    }

    seen_descriptions = set()

    fig = Figure()
    fig.update_layout(
        title="Time periods spent on different computations during rollout generation.",
        xaxis_title="time (secondss)",
        yaxis_title="rollout",
    )

    start_time = min(stat.time_init_env_started for stat in stats)
    for i_rollout, stat in enumerate(stats):
        for interval in stat.time_intervals():
            fig.add_scatter(
                x=[interval.start - start_time, interval.end - start_time],
                y=[i_rollout, i_rollout],
                name=interval.description,
                mode="lines",
                line=dict(color=description_to_color[interval.description]),
                showlegend=interval.description not in seen_descriptions,
            )
            seen_descriptions.add(interval.description)

    fig.write_html(save_filename)

    print(f"Saved plot of time periods spent on different computation during rollout generation to '{save_filename}'.")
