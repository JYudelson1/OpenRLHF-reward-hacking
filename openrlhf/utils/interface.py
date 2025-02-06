from abc import ABC, abstractmethod
from typing import *
import vllm
from vllm import SamplingParams
from dataclasses import dataclass
import ray

Message = Dict[str, str]
Reward = float
AgentState = Any  # State needed to track conversation progress

@dataclass
class AgentConversation:
    messages: List[Message]
    tokens_by_turn: List[Dict[str, Any]]

class AgentInterface(ABC):
    def __init__(
        self, 
        full_data: List[dict],
        sampling_params: SamplingParams, 
        vllm_engine: vllm.LLM, 
        **kwargs
    ):
        self.num_envs = len(full_data)
        self.full_data = full_data
        self.sampling_params = sampling_params
        self.vllm_engine = vllm_engine
        
        # As an example of full_data, for a given swe_bench task, it is a list of dicts, each with the following keys:
        # "repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement", "hints_text", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"
    
    def generate_many(self) -> List[Tuple[AgentConversation, Reward]]:
        # Initialize states for all conversations
        states = [self.init_state(data) for data in self.full_data]
        all_messages = [list() for _ in range(self.num_envs)]
        active_indices = list(range(self.num_envs))
        
        tokens_by_turn = [list() for _ in range(self.num_envs)]
        
        # Continue until all conversations are complete
        while active_indices:
            # Get next prompts for all active conversations
            all_prompts, all_states = ray.get([self.get_next_prompt_remote.remote(all_messages[idx], states[idx]) for idx in active_indices])
            active_conversations = []
            for idx in active_indices:
                #TODO: PARALLELIZE!!!
                prompt, states[idx] = all_prompts[idx], all_states[idx]
                if prompt is None:
                    # The environment is done, so we don't need to generate any more prompts
                    active_indices.remove(idx)
                    continue
                all_messages[idx].append(prompt)
                active_conversations.append(all_messages[idx])
            
            
            # Batch generate responses
            # TODO: Maybe use their tool API instead of handrolling?
            #  DEBUG ASSERT
            outputs = self.vllm_engine.chat(
                messages=active_conversations,
                sampling_params=self.sampling_params
            )
            
            # Process outputs and update states
            all_is_done = ray.get([self.is_done_remote.remote(all_messages[idx], states[idx]) for idx in active_indices])
            new_active_indices = []
            for i, output in enumerate(outputs):
                input_tokens = output.prompt_token_ids
                output_tokens = output.outputs[0].token_ids
                output_message = {"role": "assistant", "content": output.outputs[0].text}
                real_idx = active_indices[i]
                all_messages[real_idx].append(output_message)
                tokens_by_turn[real_idx].append({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                })
                if not all_is_done[i]:
                    new_active_indices.append(real_idx)
            
            active_indices = new_active_indices
        # Calculate rewards for completed conversations
        results = []
        all_rewards = ray.get([self.get_reward_remote.remote(all_messages[idx], states[idx]) for idx in active_indices])
        for i, (messages, tokens_by_turn) in enumerate(zip(all_messages, tokens_by_turn)):
            reward = all_rewards[i]
            conversation = AgentConversation(messages=messages, tokens_by_turn=tokens_by_turn)
            results.append((conversation, reward))
        
        return results

    @abstractmethod
    def init_state(self, data: dict) -> AgentState:
        """Initialize the state for a new RL env, given a dict of the info in one element of the dataset"""
        pass

    @abstractmethod
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Optional[Tuple[Message, AgentState]]:
        """Input:
        - messages: the messages in the conversation
        - state: the state of the environment
        
        Output:
        - next_prompt: the next prompt to send to the model
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
    
    @ray.remote
    def get_reward_remote(self, messages: List[Message], state: AgentState) -> Reward:
        return self.get_reward(messages, state)
    
    @ray.remote
    def is_done_remote(self, messages: List[Message], state: AgentState) -> bool:
        return self.is_done(messages, state)
    
    @ray.remote
    def get_next_prompt_remote(self, messages: List[Message], state: AgentState) -> Optional[Tuple[Message, AgentState]]:
        return self.get_next_prompt(messages, state)
    