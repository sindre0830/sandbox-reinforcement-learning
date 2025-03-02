import interface
import typing
import numpy as np


class IGameState(interface.Interface):
    def get_legal_actions(self) -> np.ndarray:
        raise NotImplementedError()

    def perform_action(self, action: typing.Tuple[int, int]) -> "IGameState":
        raise NotImplementedError()

    def is_terminal(self) -> bool:
        raise NotImplementedError()

    def get_reward(self) -> float:
        raise NotImplementedError()

    def check_winner(self) -> int:
        raise NotImplementedError()

    def next_player(self) -> int:
        raise NotImplementedError()

    def encode_state(self) -> np.ndarray:
        raise NotImplementedError()
