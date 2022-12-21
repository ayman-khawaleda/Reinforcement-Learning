from abc import ABC,abstractmethod
class Agent(ABC):
    def __init__(self, actions) -> None:
        self.actions = actions
        self.n_actions = len(self.actions)

class GridAgent(Agent):
    def __init__(self, x=0, y=0, actions=["up", "down", "left", "right"]) -> None:
        super().__init__(actions)
        self.pos = (x, y)
