import sandbox_rl.core.interfaces
import sandbox_rl.core.constants
import interface
import typing
import numpy as np
import copy


class TicTacToe(interface.implements(sandbox_rl.core.interfaces.IGameState)):
    def __init__(
        self,
        board: np.ndarray = np.zeros((3, 3), dtype=int),
        initial_player: int = sandbox_rl.core.constants.PLAYER_1,
        current_player: int = sandbox_rl.core.constants.PLAYER_1
    ) -> None:
        self.board = board
        self.initial_player = initial_player
        self.current_player = current_player

    def get_legal_actions(self) -> np.ndarray:
        return np.argwhere(self.board == sandbox_rl.core.constants.EMPTY)

    def perform_action(self, action: typing.Tuple[int, int]) -> sandbox_rl.core.interfaces.IGameState:
        new_state: TicTacToe = copy.deepcopy(self)

        row, col = action
        if new_state.board[row, col] != sandbox_rl.core.constants.EMPTY:
            raise ValueError("cell is already occupied")

        new_state.board[row, col] = new_state.current_player
        new_state.current_player = new_state.next_player()

        return new_state

    def is_terminal(self) -> bool:
        return self.check_winner() != sandbox_rl.core.constants.DNF

    def get_reward(self) -> float:
        winner = self.check_winner()

        match winner:
            case sandbox_rl.core.constants.TIE:
                return 0.0
            case self.initial_player:
                return 1.0
            case _:
                return -1.0

    def check_winner(self) -> int:
        diag_board = np.diag(self.board)
        diag_board_flipped = np.diag(np.fliplr(self.board))

        for player in [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2]:
            if (
                (self.board == player).all(axis=1).any()
                or (self.board == player).all(axis=0).any()
                or (diag_board == player).all()
                or (diag_board_flipped == player).all()
            ):
                return player

        if (self.board == sandbox_rl.core.constants.EMPTY).any():
            return sandbox_rl.core.constants.DNF

        return sandbox_rl.core.constants.TIE

    def next_player(self) -> int:
        return self.current_player ^ 3

    def encode_state(self) -> np.ndarray:
        return self.board.flatten()

    def __str__(self) -> str:
        def cell_str(cell: int) -> str:
            match cell:
                case sandbox_rl.core.constants.PLAYER_1:
                    return "X"
                case sandbox_rl.core.constants.PLAYER_2:
                    return "O"
                case _:
                    return " "

        rows: typing.List[str] = ["|".join(cell_str(cell) for cell in row) for row in self.board]

        return "\n-----\n".join(rows)
