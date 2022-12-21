from abc import ABC,abstractmethod
class Agent(ABC):
    def __init__(self, actions) -> None:
        self.actions = actions
        self.n_actions = len(self.actions)

