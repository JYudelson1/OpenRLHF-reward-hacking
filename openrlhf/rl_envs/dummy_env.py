from openrlhf.utils.interface import AgentInterface
from typing import *

type Message = Dict[str, str]
type AgentState = Any

class DummyEnv(AgentInterface):
    """This dummy environment is used for testing the RLHF pipeline.
    It's a simple environment where the agent is given a prompt and must respond to it.
    The reward is the negative sum of the lengths of the first and second responses, so we should train for shorter responses."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_state(self, data: dict) -> AgentState:
        return []
    
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Tuple[Message, AgentState]:
        if len(messages) == 0:
            turn_1_convo = self.full_data["input_prompt"]
            return turn_1_convo, []
        elif len(messages) == 2:
            turn_2_convo = {"role": "user", "content": "Okay, but what's that times five?"}
            return turn_2_convo, [messages[1]]
        else:
            raise ValueError("DummyEnv only supports 2 usermessages")
    
    def is_done(self, messages: List[Message], state: AgentState) -> bool:
        return len(messages) >=4
    
    def get_reward(self, messages: List[Message], state: AgentState) -> float:
        len_first_response = len(messages[1]["content"])
        len_second_response = len(messages[3]["content"])
        return float(len_second_response + len_first_response) * -0.01