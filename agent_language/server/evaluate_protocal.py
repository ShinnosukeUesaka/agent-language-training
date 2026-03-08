
from dataclasses import dataclass


@dataclass
class Reward:
    task_completion: bool
    total_tokens: float
    communication_tokens: float


class Agent:
    def __init__(self, model_name):
        self.model = model_name
        self.conversation_history = []
        self.client = None

    def call(self, new_message):
        self.conversation_history.append(
            {"role": "user", "message": new_message}
        )

        response  =  self.client(self.conversation_history)

        self.conversation_history.append(
            {"role": "user", "message": new_message}
        )

def evaluate_protocal(language_specification, seed) -> Reward:
    pass
    """initialize user schedule, random task, and stuff based on the seed"""