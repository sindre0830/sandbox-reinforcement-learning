import sandbox_rl.core.interfaces
import numpy as np
import interface


class RandomAgent(interface.implements(sandbox_rl.core.interfaces.IGameAgent)):
    def __init__(self, seed: int = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, game_state: sandbox_rl.core.interfaces.IGameState) -> np.ndarray:
        return self.rng.choice(game_state.get_legal_actions())
