from abc import ABC, abstractmethod
from typing import *
import vllm
from vllm import SamplingParams
from dataclasses import dataclass
import ray
import logging
import os
import pymongo
from pymongo import MongoClient
from datetime import datetime

logger = logging.getLogger(__name__)

Message = Dict[str, str]
Reward = float
AgentState = Any  # State needed to track conversation progress

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
        vllm_engine: vllm.LLM, 
        mongo_uri: Optional[str] = None,
        mongo_db_name: Optional[str] = None,
        mongo_collection_name: Optional[str] = None,
        **kwargs
    ):
        self.num_envs = len(full_data)
        self.full_data = full_data
        self.sampling_params = sampling_params
        self.vllm_engine = vllm_engine
        self.mongo_uri = mongo_uri
        self.mongo_db_name = mongo_db_name
        self.mongo_collection_name = mongo_collection_name
        
        # Check if MongoDB configuration is partially provided
        mongo_params = [mongo_uri, mongo_db_name, mongo_collection_name]
        if any(mongo_params) and not all(mongo_params):
            logger.error("MongoDB configuration is incomplete. Please provide all three parameters: "
                         "mongo_uri, mongo_db_name, and mongo_collection_name.")
        
        # As an example of full_data, for a given swe_bench task, it is a list of dicts, each with the following keys:
        # "repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement", "hints_text", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"
    
    def generate_many(self) -> List[Tuple[AgentConversation, Reward]]:
        # Initialize states for all conversations
        vllm_engine = self.vllm_engine
        self.vllm_engine = None
        states = ray.get([init_state_remote.remote(self, data) for data in self.full_data])
        self.vllm_engine = vllm_engine
        
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
                vllm_engine = self.vllm_engine
                self.vllm_engine = None
                all_prompts_and_states = ray.get([
                    get_next_prompt_remote.remote(self, messages=all_messages[idx], state=states[idx]) 
                    for idx in active_indices
                ])
                self.vllm_engine = vllm_engine
            except Exception as e:
                self.vllm_engine = vllm_engine  # Restore in case of error
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

            # Batch generate responses
            # TODO: Maybe use their tool API instead of handrolling?
            outputs = self.vllm_engine.chat(
                messages=active_conversations,
                sampling_params=self.sampling_params
            )
            
            # Process outputs and update states
            vllm_engine = self.vllm_engine
            self.vllm_engine = None
            all_is_done = ray.get([
                is_done_remote.remote(self, messages=all_messages[idx], state=states[idx]) 
                for idx in active_indices
            ])
            self.vllm_engine = vllm_engine
            new_active_indices = []
            for i, output in enumerate(outputs):
                real_idx = active_indices[i]
                if total_tokens[real_idx] == 0:
                    first_prompt_tokens[real_idx] = output.prompt_token_ids

                input_tokens = output.prompt_token_ids[total_tokens[real_idx]:]
                output_tokens = output.outputs[0].token_ids
                
                generation_starter_text = output.prompt[-10:]
                if "think" in generation_starter_text.lower():
                    output_message = {"role": "assistant", "content": "<think>" + output.outputs[0].text}
                else:
                    output_message = {"role": "assistant", "content": output.outputs[0].text}

                all_messages[real_idx].append(output_message)
                tokens_by_turn[real_idx].append({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                })
                total_tokens[real_idx] += len(input_tokens) + len(output_tokens)
                
                all_tokens[real_idx] = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                all_tokens_text[real_idx] = output.prompt + output.outputs[0].text
                if not all_is_done[i]:
                    new_active_indices.append(real_idx)
            
            active_indices = new_active_indices

        # Calculate rewards for completed conversations
        results = []
        vllm_engine = self.vllm_engine
        self.vllm_engine = None
        all_rewards = ray.get([
            get_reward_remote.remote(self, messages=all_messages[idx], state=states[idx]) 
            for idx in range(self.num_envs)
        ])
        self.vllm_engine = vllm_engine
        
        # Create results list
        results_data = []
        for i, (messages, tokens_by_turn_one_env, fpt, aot) in enumerate(zip(all_messages, tokens_by_turn, first_prompt_tokens, all_tokens)):
            reward = all_rewards[i]
            conversation = AgentConversation(messages=messages, tokens_by_turn=tokens_by_turn_one_env, first_prompt_tokens=fpt, all_tokens=aot)
            results.append((conversation, reward))
            
            # Prepare data for MongoDB upload
            results_data.append({
                "messages": messages,
                "all_text": all_tokens_text[i],
                "reward": float(reward),
                "task_prompt": messages[0]["content"],
            })
        
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
        
        return results

    @abstractmethod
    def init_state(self, data: dict) -> AgentState:
        """Initialize the state for a new RL env, given a dict of the info in one element of the dataset"""
        pass

    @abstractmethod
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Optional[Tuple[Union[List[Message], Message], AgentState]]:
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
def get_next_prompt_remote(agent: AgentInterface, messages: List[Message], state: AgentState) -> Optional[Tuple[Union[List[Message], Message], AgentState]]:
    return agent.get_next_prompt(messages, state)