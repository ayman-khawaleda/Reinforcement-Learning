from .Agent import Agent
class GridAgent(Agent):
    def __init__(self, x=0, y=0, actions=["up", "down", "left", "right"]) -> None:
        super().__init__(actions)
        self.pos = (x, y)
