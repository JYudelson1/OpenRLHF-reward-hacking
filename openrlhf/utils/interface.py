from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter
from typing import *
import ray.remote_function
import vllm
from vllm import CompletionOutput, SamplingParams, RequestOutput
from openai import OpenAI
from anthropic import Anthropic
import json
import ray
import logging
from dataclasses import dataclass
from pymongo import MongoClient
from datetime import datetime

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


class AgentInterface(ABC):
    def __init__(
        self,
        full_data: List[dict],
        sampling_params: SamplingParams,
        llm_engine: vllm.LLM | OpenAI | Anthropic,
        mongo_uri: Optional[str] = None,
        mongo_db_name: Optional[str] = None,
        mongo_collection_name: Optional[str] = None,
        environment_parallelism: Literal["ray", "threading"] = "threading",
        openai_or_anthropic_model: str | None = None,
        anthropic_thinking: Any = None,
        **kwargs,
    ):
        self.num_envs = len(full_data)
        self.full_data = full_data
        self.sampling_params = sampling_params
        self.llm_engine = llm_engine
        self.mongo_uri = mongo_uri
        self.mongo_db_name = mongo_db_name
        self.mongo_collection_name = mongo_collection_name
        self.environment_parallelism = environment_parallelism
        self.openai_or_anthropic_model = openai_or_anthropic_model
        self.anthropic_thinking = anthropic_thinking

        # Check if MongoDB configuration is partially provided
        mongo_params = [mongo_uri, mongo_db_name, mongo_collection_name]
        if any(mongo_params) and not all(mongo_params):
            logger.error(
                "MongoDB configuration is incomplete. Please provide all three parameters: "
                "mongo_uri, mongo_db_name, and mongo_collection_name."
            )

        # As an example of full_data, for a given swe_bench task, it is a list of dicts, each with the following keys:
        # "repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement", "hints_text", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"

    def generate_many(self) -> List[Tuple[AgentConversation, Reward]]:
        everything_start_time = perf_counter()

        times_doing_environment_steps = []
        times_generating_completions = []
        times_evaluating_is_done = []

        # Initialize states for all conversations
        llm_engine = self.llm_engine
        self.llm_engine = None

        init_env_start_time = perf_counter()
        # states = ray.get([init_state_remote.remote(self, data) for data in self.full_data])
        states = self.run_environment_calls_in_parallel(
            DelayedFunction(
                function=self.__class__.init_state,
                remote_function=init_state_remote,
                args=(self,),
                kwargs={"data": data},
            )
            for data in self.full_data
        )
        init_env_end_time = perf_counter()
        time_initializing_environments = init_env_end_time - init_env_start_time

        self.llm_engine = llm_engine

        all_messages = [list() for _ in range(self.num_envs)]
        active_indices = list(range(self.num_envs))

        tokens_by_turn = [list() for _ in range(self.num_envs)]
        total_tokens = [0 for _ in range(self.num_envs)]
        first_prompt_tokens = [None for _ in range(self.num_envs)]
        all_tokens = [[] for _ in range(self.num_envs)]
        all_tokens_text = [[] for _ in range(self.num_envs)]
        # Continue until all conversations are complete
        while active_indices:
            # Get next prompts for all active conversations
            try:
                # Temporarily remove vllm engine before sending through Ray
                llm_engine = self.llm_engine
                self.llm_engine = None

                env_step_start_time = perf_counter()
                # all_prompts_and_states = ray.get(
                #    [
                #         get_next_prompt_remote.remote(self, messages=all_messages[idx], state=states[idx])
                #         for idx in active_indices
                #     ]
                # )
                all_prompts_and_states = self.run_environment_calls_in_parallel(
                    DelayedFunction(
                        function=self.__class__.get_next_prompt,
                        remote_function=get_next_prompt_remote,
                        args=(self,),
                        kwargs={"messages": all_messages[idx], "state": states[idx]},
                    )
                    for idx in active_indices
                )
                env_step_end_time = perf_counter()
                times_doing_environment_steps.append(env_step_end_time - env_step_start_time)

                self.llm_engine = llm_engine
            except Exception as e:
                self.llm_engine = llm_engine  # Restore in case of error
                logger.error(f"Error getting prompts: {str(e)}")
                raise

            active_conversations = []
            indices_to_remove = []
            for i, idx in enumerate(active_indices):
                result = all_prompts_and_states[i]
                if result is None:
                    active_indices.remove(idx)
                    continue

                prompt, states[idx] = result
                if prompt is None or states[idx] is None:
                    # The environment is done, so we don't need to generate any more prompts
                    # active_indices.remove(idx)
                    indices_to_remove.append(idx)
                    continue
                if isinstance(prompt, list):
                    all_messages[idx].extend(prompt)
                elif isinstance(prompt, dict):
                    all_messages[idx].append(prompt)
                else:
                    raise ValueError(f"Invalid prompt type: {type(prompt)}")
                active_conversations.append(all_messages[idx])

            for idx in indices_to_remove:
                active_indices.remove(idx)

            # Leave the loop if all conversations are done
            if len(active_conversations) == 0:
                break

            generate_start_time = perf_counter()
            # Batch generate responses
            # TODO: Maybe use their tool API instead of handrolling?
            outputs = self._generate_chat_completions(active_conversations)
            # outputs = self.llm_engine.chat(messages=active_conversations, sampling_params=self.sampling_params)
            generate_end_time = perf_counter()
            times_generating_completions.append(generate_end_time - generate_start_time)

            # Process outputs and update states
            llm_engine = self.llm_engine
            self.llm_engine = None

            is_done_start_time = perf_counter()
            # all_is_done = ray.get(
            #     [is_done_remote.remote(self, messages=all_messages[idx], state=states[idx]) for idx in active_indices]
            # )
            all_is_done = self.run_environment_calls_in_parallel(
                DelayedFunction(
                    function=self.__class__.is_done,
                    remote_function=is_done_remote,
                    args=(self,),
                    kwargs={"messages": all_messages[idx], "state": states[idx]},
                )
                for idx in active_indices
            )
            is_done_end_time = perf_counter()
            times_evaluating_is_done.append(is_done_end_time - is_done_start_time)

            self.llm_engine = llm_engine
            new_active_indices = []
            for i, output in enumerate(outputs):
                real_idx = active_indices[i]
                if total_tokens[real_idx] == 0:
                    first_prompt_tokens[real_idx] = output.prompt_token_ids

                input_tokens = output.prompt_token_ids[total_tokens[real_idx] :]
                output_tokens = output.outputs[0].token_ids

                generation_starter_text = output.prompt[-10:]
                if "think" in generation_starter_text.lower():
                    output_message = {"role": "assistant", "content": "<think>" + output.outputs[0].text}
                else:
                    output_message = {"role": "assistant", "content": output.outputs[0].text}

                all_messages[real_idx].append(output_message)
                tokens_by_turn[real_idx].append({"input_tokens": input_tokens, "output_tokens": output_tokens})
                total_tokens[real_idx] += len(input_tokens) + len(output_tokens)

                all_tokens[real_idx] = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                all_tokens_text[real_idx] = output.prompt + output.outputs[0].text
                if not all_is_done[i]:
                    new_active_indices.append(real_idx)

            active_indices = new_active_indices

        # Calculate rewards for completed conversations
        results = []
        llm_engine = self.llm_engine
        self.llm_engine = None

        rewards_start_time = perf_counter()
        # all_rewards = ray.get(
        #     [
        #         get_reward_remote.remote(self, messages=all_messages[idx], state=states[idx])
        #         for idx in range(self.num_envs)
        #     ]
        # )
        all_rewards = self.run_environment_calls_in_parallel(
            DelayedFunction(
                function=self.__class__.get_reward,
                remote_function=get_reward_remote,
                args=(self,),
                kwargs={"messages": all_messages[idx], "state": states[idx]},
            )
            for idx in range(self.num_envs)
        )
        rewards_end_time = perf_counter()
        time_computing_rewards = rewards_end_time - rewards_start_time

        self.llm_engine = llm_engine

        # Create results list
        results_data = []
        for i, (messages, tokens_by_turn_one_env, fpt, aot) in enumerate(
            zip(all_messages, tokens_by_turn, first_prompt_tokens, all_tokens)
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
                    "all_text": all_tokens_text[i],
                    "reward": float(reward),
                    "task_prompt": messages[0]["content"],
                }
            )

        # Upload results to MongoDB after all processing is complete
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

        everything_end_time = perf_counter()
        total_time = everything_end_time - everything_start_time

        logger.info(f"Rollout completed in {int(total_time)} seconds. Breakdown of time spent:")
        logger.info(
            f"Generating completions with vllm: {int(sum(times_generating_completions))} seconds ({sum(times_generating_completions) / total_time:.0%}, breakdown by step: {', '.join(str(int(t)) for t in times_generating_completions)})"
        )
        logger.info(
            f"Initializing environments: {int(time_initializing_environments)} seconds ({time_initializing_environments / total_time:.0%})"
        )
        logger.info(
            f"Doing environment steps: {int(sum(times_doing_environment_steps))} seconds ({sum(times_doing_environment_steps) / total_time:.0%}, breakdown by step: {', '.join(str(int(t)) for t in times_doing_environment_steps)})"
        )
        logger.info(
            f"Evaluating whether environments are done: {int(sum(times_evaluating_is_done))} seconds ({sum(times_evaluating_is_done) / total_time:.0%}, breakdown by step: {', '.join(str(int(t)) for t in times_evaluating_is_done)})"
        )
        logger.info(f"Computing rewards: {int(time_computing_rewards)} ({time_computing_rewards / total_time:.0%})")
        unaccounted_for_time = (
            total_time
            - sum(times_generating_completions)
            - time_initializing_environments
            - sum(times_doing_environment_steps)
            - sum(times_evaluating_is_done)
            - time_computing_rewards
        )
        logger.info(
            f"Unaccounted for (should be close to zero): {int(unaccounted_for_time)} ({unaccounted_for_time / total_time:.0%})"
        )

        return results

    def _generate_chat_completions(self, messages: list[list[Message]]) -> list[RequestOutput]:
        if isinstance(self.llm_engine, vllm.LLM):
            return self._generate_chat_completions_vllm(messages)
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

            if len(merged_messages[-1]) > 0 or message["role"] == merged_messages[-1]["role"]:
                assert set(message.keys()) == {"role", "content"}
                merged_messages[-1]["content"] += "\n\n" + message["content"]
                continue

            merged_messages.append(message)

        return merged_messages

    def _generate_chat_completions_vllm(self, messages: list[list[Message]]) -> list[RequestOutput]:
        return self.llm_engine.chat(messages=messages, sampling_params=self.sampling_params)  # type: ignore

    def _generate_chat_completions_openai(self, messages: list[list[Message]]) -> list[RequestOutput]:
        assert self.openai_or_anthropic_model is not None, (
            "AgentInterface.openai_or_anthropic_model should be provided on initialization if AgentInterface.llm_engine is of type OpenAI."
        )

        def single_completion(conversation: list[Message]) -> RequestOutput:
            conversation = self._merge_tool_and_user_messages(conversation)

            completion = self.llm_engine.chat.completions.create(  # type: ignore
                messages=conversation,  # type: ignore
                model=self.openai_or_anthropic_model,  # type: ignore
                temperature=self.sampling_params.temperature,
                max_completion_tokens=self.sampling_params.max_tokens,
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
        pass

    def run_environment_calls_in_parallel(self, calls: Iterable[DelayedFunction]) -> list[Any]:
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
            f"Invalid value `{self.environment_parallelism}` of AgentInterface.environment_parallelism. It should be either 'ray' or 'threading'."
        )


@ray.remote
def init_state_remote(agent: AgentInterface, data: dict) -> AgentState:
    return agent.init_state(data)


@ray.remote
def get_reward_remote(agent: AgentInterface, messages: List[Message], state: AgentState) -> Reward:
    return agent.get_reward(messages, state)


@ray.remote
def is_done_remote(agent: AgentInterface, messages: List[Message], state: AgentState) -> bool:
    return agent.is_done(messages, state)


@ray.remote
def get_next_prompt_remote(
    agent: AgentInterface, messages: List[Message], state: AgentState
) -> Optional[Tuple[Union[List[Message], Message], AgentState]]:
    return agent.get_next_prompt(messages, state)
