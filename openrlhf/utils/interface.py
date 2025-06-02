import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter
from typing import *
from uuid import uuid4
import ray.remote_function
import vllm
from vllm import CompletionOutput, SamplingParams
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
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
        logger.info(f"Computing rewards: {int(self.time_computing_rewards)} ({self.time_computing_rewards / self.total_time:.0%})")
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
    
class AgentInterface(ABC):
    def __init__(
        self,
        full_data: List[dict],
        sampling_params: SamplingParams,
        llm_engine: vllm.AsyncLLMEngine | OpenAI | Anthropic,
        async_event_loop: asyncio.AbstractEventLoop,
        mongo_uri: Optional[str] = None,
        mongo_db_name: Optional[str] = None,
        mongo_collection_name: Optional[str] = None,
        environment_parallelism: Literal["ray", "threading"] | None = "threading",
        openai_or_anthropic_model: str | None = None,
        anthropic_thinking: Any = None,
        truncate_prompt_tokens: Optional[int] = None,
        stop_on_truncation: bool = False,
        length_penalty: float = 0.0,
        stop_strings: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ):
        self.num_envs = len(full_data)
        self.full_data = full_data
        self.sampling_params = sampling_params
        self.llm_engine = llm_engine
        self.async_event_loop = async_event_loop
        self.mongo_uri = mongo_uri
        self.mongo_db_name = mongo_db_name
        self.mongo_collection_name = mongo_collection_name
        self.environment_parallelism = environment_parallelism
        self.openai_or_anthropic_model = openai_or_anthropic_model
        self.anthropic_thinking = anthropic_thinking
        self.tokenizer = None
        self.stop_on_truncation = stop_on_truncation
        self.length_penalty = length_penalty
        self.stop_strings = stop_strings
        self.max_steps = max_steps
        
        if stop_strings is not None:
            self.sampling_params.stop = stop_strings
            self.sampling_params.include_stop_str_in_output = True
        
        self.all_messages = [list() for _ in range(self.num_envs)]
        self.active_indices = list(range(self.num_envs))
        self.tokens_by_turn = [list() for _ in range(self.num_envs)]
        self.total_tokens = [0 for _ in range(self.num_envs)]
        self.first_prompt_tokens = [None for _ in range(self.num_envs)]
        self.all_tokens = [[] for _ in range(self.num_envs)]
        self.num_steps = [0 for _ in range(self.num_envs)]
        
        # Set truncate_prompt_tokens in sampling_params if provided
        if truncate_prompt_tokens is not None:
            if isinstance(self.llm_engine, vllm.LLM):
                # Set it in SamplingParams for vLLM
                self.sampling_params.truncate_prompt_tokens = truncate_prompt_tokens
                logger.info(f"Set SamplingParams.truncate_prompt_tokens to {truncate_prompt_tokens}")
            else:
                logger.warning(
                    f"truncate_prompt_tokens ({truncate_prompt_tokens}) provided but LLM engine is not vLLM. "
                    "This parameter will be ignored for both SamplingParams and manual truncation."
                )

        # Check if MongoDB configuration is partially provided
        mongo_params = [mongo_uri, mongo_db_name, mongo_collection_name]
        if any(mongo_params) and not all(mongo_params):
            logger.error(
                "MongoDB configuration is incomplete. Please provide all three parameters: "
                "mongo_uri, mongo_db_name, and mongo_collection_name."
            )

        # As an example of full_data, for a given swe_bench task, it is a list of dicts, each with the following keys:
        # "repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement", "hints_text", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"
        
    @abstractmethod
    def init_state(self, data: dict) -> AgentState:
        """Initialize the state for a new RL env, given a dict of the info in one element of the dataset"""
        pass

    @abstractmethod
    def get_next_prompt(
        self, messages: List[Message], state: AgentState
    ) -> Optional[Tuple[Union[List[Message], Message], AgentState]]:
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
    
    ### Below are the logical pieces

    def generate_many(self) -> List[Tuple[AgentConversation, Reward]]:
        ## Initialize time metrics
        self.time_metrics = TimeMetrics()

        ## Initialize states for all conversations
        self.states = self._init_all_states()

        # Continue until all conversations are complete
        while self.active_indices:
            # Get next prompts for all active conversations
            all_prompts_and_states = self._step_through_all_envs()

            active_conversations = self._process_next_prompts(all_prompts_and_states)
            
            # Leave the loop if all conversations are done
            if len(active_conversations) == 0:
                break

            x = self._generate_all_chat_completions(active_conversations)
            print(f"{x=}")
            outputs, was_truncated = x

            # Process outputs and update states
            all_is_done = self._check_all_done()
            
            self._process_outputs(outputs, was_truncated, all_is_done)

        # Calculate rewards for completed conversations
        results, results_data = self._calculate_rewards()

        # Upload results to MongoDB after all processing is complete
        self.log_rollouts_mongodb(results_data)

        self.time_metrics.finish_and_log()
        
        return results
    
    def _init_all_states(self) -> list[AgentState]:
        ## Initialize states for all conversations (remove llm engine before sending through Ray)
        llm_engine = self.llm_engine
        self.llm_engine = None

        self.time_metrics.init_env_start_time = perf_counter()
        states = self.run_environment_calls_in_parallel(
            DelayedFunction(
                function=self.__class__.init_state,
                remote_function=init_state_remote,
                args=(self,),
                kwargs={"data": data},
            )
            for data in self.full_data
        )
        self.time_metrics.time_initializing_environments = perf_counter() - self.time_metrics.init_env_start_time

        # Restore llm engine
        self.llm_engine = llm_engine

        return states
    
    def _step_through_all_envs(self) -> list[Tuple[Optional[Tuple[List[Message], AgentState]], Optional[Tuple[List[Message], AgentState]]]]:
        try:
            # Temporarily remove vllm engine before sending through Ray
            llm_engine = self.llm_engine
            self.llm_engine = None

            env_step_start_time = perf_counter()

            all_prompts_and_states = self.run_environment_calls_in_parallel(
                DelayedFunction(
                    function=self.__class__.get_next_prompt,
                    remote_function=get_next_prompt_remote,
                    args=(self,),
                    kwargs={"messages": self.all_messages[idx], "state": self.states[idx]},
                )
                for idx in self.active_indices
            )
            env_step_end_time = perf_counter()
            self.time_metrics.time_doing_environment_steps.append(env_step_end_time - env_step_start_time)

            self.llm_engine = llm_engine
        except Exception as e:
            self.llm_engine = llm_engine  # Restore in case of error
            logger.error(f"Error getting prompts: {str(e)}")
            raise
        
        return all_prompts_and_states
    
    def _process_next_prompts(self, all_prompts_and_states: list[Tuple[Optional[Tuple[List[Message], AgentState]], Optional[Tuple[List[Message], AgentState]]]]) -> list[AgentConversation]:
        active_conversations = []
        indices_to_remove = []
        for i, idx in enumerate(self.active_indices):
            result = all_prompts_and_states[i]
            if result is None:
                self.active_indices.remove(idx)
                continue

            prompt, self.states[idx] = result
            if prompt is None or self.states[idx] is None:
                # The environment is done, so we don't need to generate any more prompts
                # active_indices.remove(idx)
                indices_to_remove.append(idx)
                continue
            if isinstance(prompt, list):
                self.all_messages[idx].extend(prompt)
            elif isinstance(prompt, dict):
                self.all_messages[idx].append(prompt)
            else:
                raise ValueError(f"Invalid prompt type: {type(prompt)}")
            active_conversations.append(self.all_messages[idx])

        for idx in indices_to_remove:
            self.active_indices.remove(idx)
            
        return active_conversations
    
    def _generate_all_chat_completions(self, active_conversations: list[list[Message]]) -> Tuple[list[RequestOutput], List[bool]]:
        generate_start_time = perf_counter()
        # Batch generate responses
        # TODO: Maybe use their tool API instead of handrolling?
        if not isinstance(self.llm_engine, (OpenAI, Anthropic)) and self.stop_on_truncation:
            outputs, was_truncated = self._generate_chat_completions(active_conversations)
        else:
            outputs = self._generate_chat_completions(active_conversations)
            was_truncated = [False] * len(outputs)
        # outputs = self.llm_engine.chat(messages=active_conversations, sampling_params=self.sampling_params)
        generate_end_time = perf_counter()
        self.time_metrics.time_generating_completions.append(generate_end_time - generate_start_time)
        
        print(f"{outputs=} {was_truncated=}")
        return outputs, was_truncated
    
    def _check_all_done(self) -> list[bool]:
        llm_engine = self.llm_engine
        self.llm_engine = None

        is_done_start_time = perf_counter()

        all_is_done = self.run_environment_calls_in_parallel(
            DelayedFunction(
                function=self.__class__.is_done,
                remote_function=is_done_remote,
                args=(self,),
                kwargs={"messages": self.all_messages[idx], "state": self.states[idx]},
            )
            for idx in self.active_indices
        )
        is_done_end_time = perf_counter()
        
        self.time_metrics.time_evaluating_is_done.append(is_done_end_time - is_done_start_time)
        self.llm_engine = llm_engine
        
        return all_is_done
            
    def _process_outputs(self, outputs: list[RequestOutput], was_truncated: list[bool], all_is_done: list[bool]) -> None:
        new_active_indices = []
        for i, output in enumerate(outputs):
            real_idx = self.active_indices[i]
            if self.total_tokens[real_idx] == 0:
                self.first_prompt_tokens[real_idx] = output.prompt_token_ids

            input_tokens = output.prompt_token_ids[self.total_tokens[real_idx] :]
            output_tokens = output.outputs[0].token_ids

            # generation_starter_text = output.prompt[-10:]
            # if "think" in generation_starter_text.lower():
            #     output_message = {"role": "assistant", "content": "<think>" + output.outputs[0].text}
            # else:
            #     output_message = {"role": "assistant", "content": output.outputs[0].text}
            output_message = {"role": "assistant", "content": output.outputs[0].text}
            self.all_messages[real_idx].append(output_message)
            self.tokens_by_turn[real_idx].append({"input_tokens": input_tokens, "output_tokens": output_tokens})
            self.total_tokens[real_idx] += len(input_tokens) + len(output_tokens)

            self.all_tokens[real_idx] = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
            #all_tokens_text[real_idx] = output.prompt + output.outputs[0].text
            
            # Stop reason: truncation
            if self.stop_on_truncation and was_truncated[i]:
                all_is_done[i] = True
                
            # Stop reason: max steps
            if self.max_steps is not None:
                self.num_steps[real_idx] += 1
                if self.num_steps[real_idx] >= self.max_steps:
                    all_is_done[i] = True
                
            if not all_is_done[i]:
                new_active_indices.append(real_idx)

        self.active_indices = new_active_indices
        
    def _calculate_rewards(self) -> Tuple[list[Tuple[AgentConversation, Reward]], list[dict]]:
        results = []
        llm_engine = self.llm_engine
        self.llm_engine = None

        self.time_metrics.compute_rewards_start_time = perf_counter()

        all_rewards = self.run_environment_calls_in_parallel(
            DelayedFunction(
                function=self.__class__.get_reward,
                remote_function=get_reward_remote,
                args=(self,),
                kwargs={"messages": self.all_messages[idx], "state": self.states[idx]},
            )
            for idx in range(self.num_envs)
        )
        self.time_metrics.time_computing_rewards = perf_counter() - self.time_metrics.compute_rewards_start_time

        self.llm_engine = llm_engine

        # Create results list
        results_data = []
        for i, (messages, tokens_by_turn_one_env, fpt, aot) in enumerate(
            zip(self.all_messages, self.tokens_by_turn, self.first_prompt_tokens, self.all_tokens)
        ):
            reward = all_rewards[i]
            conversation = AgentConversation(
                messages=messages, tokens_by_turn=tokens_by_turn_one_env, first_prompt_tokens=fpt, all_tokens=aot
            )
            results.append((conversation, reward))

            # Prepare data for MongoDB upload
            results_data.append(
                {
                    "messages": messages,
                    #"all_text": all_tokens_text[i],
                    "reward": float(reward),
                    "task_prompt": messages[0]["content"],
                }
            )
        
        return results, results_data
    
    def log_rollouts_mongodb(self, results_data: list[dict]) -> None:
        mongo_params = [self.mongo_uri, self.mongo_db_name, self.mongo_collection_name]
        if all(mongo_params):
            try:
                # Connect to MongoDB
                mongo_client = MongoClient(self.mongo_uri)
                mongo_db = mongo_client[self.mongo_db_name]
                mongo_collection = mongo_db[self.mongo_collection_name]

                # Upload all results
                for i, data in enumerate(results_data):
                    # Add timestamp at upload time
                    data["timestamp"] = datetime.utcnow()
                    mongo_collection.insert_one(data)

                logger.info(f"Uploaded {len(results_data)} conversations to MongoDB")
                mongo_client.close()
            except Exception as e:
                logger.error(f"Failed to upload conversations to MongoDB: {str(e)}")

    def _generate_chat_completions(self, messages: list[list[Message]]) -> Tuple[list[RequestOutput], List[bool]]|list[RequestOutput]:
        if not isinstance(self.llm_engine, (OpenAI, Anthropic)):
             #vLLM will apply its own truncation based on sampling_params.truncate_prompt_tokens if set
            return self.async_event_loop.run_until_complete(self._vllm_chat_with_truncation(
                engine=self.llm_engine, 
                messages=messages, 
                sampling_params=self.sampling_params, 
                truncation_amt=self.sampling_params.truncate_prompt_tokens
            ))
        if isinstance(self.llm_engine, OpenAI):
            return self._generate_chat_completions_openai(messages)
        if isinstance(self.llm_engine, Anthropic):
            return self._generate_chat_completions_anthropic(messages)
        raise TypeError(
            f"AgentInterface.llm_engine should be of type vllm.LLM, OpenAI, or Anthropic, but is of type {type(self.llm_engine)}."
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
    
    async def _vllm_chat_with_truncation(
        self,
        engine: vllm.AsyncLLMEngine,
        messages: list[list[Message]],
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        truncation_amt: Optional[int] = None
    ) -> Tuple[list[RequestOutput], List[bool]]|list[RequestOutput]:
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
        list_of_messages: list[list[ChatCompletionMessageParam]] = messages

        tokenizer = await engine.get_tokenizer()
        model_config = await engine.get_model_config()
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tools,
            chat_template_content_format,
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )

        prompts: list[TokensPrompt] = []
        
        was_truncated: List[bool] = []

        for msgs in list_of_messages:
            # NOTE: _parse_chat_message_content_parts() currently doesn't
            # handle mm_processor_kwargs, since there is no implementation in
            # the chat message parsing for it.
            conversation, _ = parse_chat_messages(
                msgs,
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
            prompt_token_ids = tokenizer.encode(prompt_str,
                                                add_special_tokens=False)
            
            # Truncate the prompt if necessary
            if truncation_amt is not None and len(prompt_token_ids) > truncation_amt:
                old_len = len(prompt_token_ids)
                prompt_token_ids = prompt_token_ids[:truncation_amt]
                logger.warning(f"Truncated prompt from {old_len} tokens to {truncation_amt} tokens.")
                was_truncated.append(True)
            else:
                was_truncated.append(False)
            
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            prompts.append(prompt)

        outputs = [None] * len(prompts)

        for i, prompt in enumerate(prompts):
            async for output in engine.generate(
                prompt,
                sampling_params,
                str(uuid4()),
                # use_tqdm=use_tqdm,
                lora_request=lora_request,
            ):
                if output.finished:
                    assert outputs[i] is None
                    outputs[i] = output

        assert all(output is not None for output in outputs)
            
        if truncation_amt is not None:
            return outputs, was_truncated
        else:
            return outputs

    def _generate_chat_completions_openai(self, messages: list[list[Message]]) -> list[RequestOutput]:
        assert self.openai_or_anthropic_model is not None, (
            "AgentInterface.openai_or_anthropic_model should be provided on initialization if AgentInterface.llm_engine is of type OpenAI."
        )

        @retry(
            stop=stop_after_attempt(8),
            wait=wait_exponential(multiplier=15, min=1),
            before_sleep=lambda retry_state: print(f"Calling OpenAI API: Attempt {retry_state.attempt_number} Failed: Exception: {retry_state.outcome.exception()}", file=stderr)
        )
        def single_completion(conversation: list[Message]) -> RequestOutput:
            conversation = self._merge_tool_and_user_messages(conversation)

            completion = self.llm_engine.chat.completions.create(  # type: ignore
                messages=conversation,  # type: ignore
                model=self.openai_or_anthropic_model,  # type: ignore
                temperature=self.sampling_params.temperature,
                max_completion_tokens=self.sampling_params.max_tokens,
                stop=self.stop_strings,
            )
            return RequestOutput(
                request_id="",
                prompt="Calling openai API with the following messages: " + json.dumps(conversation) + "\n",
                prompt_token_ids=[-1] * completion.usage.prompt_tokens,  # type: ignore
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=completion.choices[0].message.content,  # type: ignore
                        token_ids=[-1] * completion.usage.completion_tokens,  # type: ignore
                        cumulative_logprob=None,
                        logprobs=None,
                    )
                ],
                finished=True,
            )

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(single_completion, conversation) for conversation in messages]
            completions = []
            for future in futures:
                completions.append(future.result())
        return completions

    def _generate_chat_completions_anthropic(self, messages: list[list[Message]]) -> list[RequestOutput]:
        assert self.openai_or_anthropic_model is not None, (
            "AgentInterface.openai_or_anthropic_model should be provided on initialization if AgentInterface.llm_engine is of type Anthropic."
        )

        @retry(
            stop=stop_after_attempt(8),
            wait=wait_exponential(multiplier=15, min=1),
            before_sleep=lambda retry_state: print(f"Calling Anthropic API: Attempt {retry_state.attempt_number} Failed: Exception: {retry_state.outcome.exception()}", file=stderr)
        )
        def single_completion(conversation: list[Message]) -> RequestOutput:
            conversation = self._merge_tool_and_user_messages(conversation)

            api_kwargs = {}

            assert len(conversation) > 0
            if conversation[0]["role"] == "system":
                assert set(conversation[0].keys()) == {"role", "content"}
                system_message = conversation[0]["content"]
                conversation = conversation[1:]
                api_kwargs["system"] = system_message

            if self.anthropic_thinking is not None:
                api_kwargs["thinking"] = self.anthropic_thinking
                
            if self.stop_strings is not None:
                api_kwargs["stop_sequences"] = self.stop_strings

            completion = self.llm_engine.messages.create(  # type: ignore
                messages=conversation,  # type: ignore
                model=self.openai_or_anthropic_model,  # type: ignore
                max_tokens=self.sampling_params.max_tokens,  # type: ignore
                temperature=self.sampling_params.temperature,
                **api_kwargs,  # type: ignore
            )

            if "system" in api_kwargs.keys():
                prompt_string = (
                    "Calling Anthropic API with the following messages:\n"
                    f"System message: {api_kwargs['system']}\n"
                    f"All other messages:\n {json.dumps(conversation)}"
                )
            else:
                prompt_string = f"Calling Anthropic API wit the following messages: {json.dumps(conversation)}"

            return RequestOutput(
                request_id="",
                prompt=prompt_string,
                prompt_token_ids=[-1] * completion.usage.input_tokens,  # type: ignore
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=completion.content[0].text,  # type: ignore
                        token_ids=[-1] * completion.usage.output_tokens,  # type: ignore
                        cumulative_logprob=None,
                        logprobs=None,
                    )
                ],
                finished=True,
            )

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(single_completion, conversation) for conversation in messages]
            completions = []
            for future in futures:
                completions.append(future.result())
        return completions

    def run_environment_calls_in_parallel(self, calls: Iterable[DelayedFunction]) -> list[Any]:
        if self.environment_parallelism is None:
            return [call.function(*call.args, **call.kwargs) for call in calls]

        if self.environment_parallelism == "ray":
            return ray.get([call.remote_function.remote(*call.args, **call.kwargs) for call in calls])

        if self.environment_parallelism == "threading":
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(lambda call: call.function(*call.args, **call.kwargs), call) for call in calls
                ]
                results = []
                for future in futures:
                    results.append(future.result())
            return results

        raise ValueError(
            f"Invalid value `{self.environment_parallelism}` of AgentInterface.environment_parallelism. It should be either 'ray', 'threading', or None."
        )
        
    def _length_penalty(self, conversation: list[Message]) -> float:
        total_assistant_message_length = sum(
            len(message["content"])
            for message in conversation
            if message["role"] == "assistant"
        )
        return total_assistant_message_length * self.length_penalty

@ray.remote
def init_state_remote(agent: AgentInterface, data: dict) -> AgentState:
    return agent.init_state(data)


@ray.remote
def get_reward_remote(agent: AgentInterface, messages: List[Message], state: AgentState) -> Reward:
    return agent.get_reward(messages, state) + agent._length_penalty(messages)


@ray.remote
def is_done_remote(agent: AgentInterface, messages: List[Message], state: AgentState) -> bool:
    return agent.is_done(messages, state)


@ray.remote
def get_next_prompt_remote(
    agent: AgentInterface, messages: List[Message], state: AgentState
) -> Optional[Tuple[Union[List[Message], Message], AgentState]]:
    return agent.get_next_prompt(messages, state)
