import sandbox_rl.core.interfaces
import numpy as np
import interface
import typing


class RandomAgent(interface.implements(sandbox_rl.core.interfaces.IGameAgent)):
    def __init__(self, seed: int = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, game_state: sandbox_rl.core.interfaces.IGameState) -> typing.Tuple[np.ndarray, float]:
        legal_actions = game_state.get_legal_actions()
        num_actions = len(legal_actions)
        probabilities = np.ones(num_actions) / num_actions

        value = 0.0

        return probabilities, value

    def train(self, batch: typing.Any) -> None:
        pass
