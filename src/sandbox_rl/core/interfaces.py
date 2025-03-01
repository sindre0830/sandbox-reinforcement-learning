import interface
import typing


class IGameState(interface.Interface):
    def get_legal_actions(self) -> typing.List[typing.Tuple[int, int]]:
        raise NotImplementedError()

    def perform_action(self, action: typing.Tuple[int, int]) -> "IGameState":
        raise NotImplementedError()

    def is_terminal(self) -> bool:
        raise NotImplementedError()

    def get_reward(self) -> float:
        raise NotImplementedError()

    def check_winner(self) -> typing.Optional[typing.Any]:
        raise NotImplementedError()

    def next_player(self) -> typing.Any:
        raise NotImplementedError()

    def deep_copy(self) -> "IGameState":
        raise NotImplementedError()
