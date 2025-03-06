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

    def __str__(self) -> str:
        raise NotImplementedError()


class IGameAgent(interface.Interface):
    def select_action(self, game_state: IGameState) -> typing.Tuple[np.ndarray, float]:
        raise NotImplementedError()

    def train(self, batch: typing.Any) -> None:
        raise NotImplementedError()


class ILearningAgent(interface.Interface):
    def execute(self) -> None:
        raise NotImplementedError()


class IReplayBuffer(interface.Interface):
    def __init__(self, max_size: int = 10000) -> None:
        raise NotImplementedError()

    def store(self, data: typing.Tuple) -> None:
        raise NotImplementedError()

    def sample(self, batch_size: int) -> typing.List[typing.Tuple]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
