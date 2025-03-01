import sandbox_rl.core.interfaces
import interface
import typing


class TicTacToe(interface.implements(sandbox_rl.core.interfaces.IGameState)):
    def __init__(
        self,
        board: typing.Optional[typing.List[typing.List[typing.Optional[str]]]] = None,
        initial_player: str = "X",
        current_player: str = "X"
    ) -> None:
        if board is None:
            board = [[None for _ in range(3)] for _ in range(3)]

        self.board = board
        self.initial_player = initial_player
        self.current_player = current_player

    def get_legal_actions(self) -> typing.List[typing.Tuple[int, int]]:
        actions: typing.List[typing.Tuple[int, int]] = []

        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    actions.append((row, col))

        return actions

    def perform_action(self, action: typing.Tuple[int, int]) -> sandbox_rl.core.interfaces.IGameState:
        new_state: TicTacToe = self.deep_copy()

        row, col = action
        if new_state.board[row][col] is not None:
            raise ValueError("cell is already occupied")

        new_state.board[row][col] = new_state.current_player
        new_state.current_player = new_state.next_player()

        return new_state

    def is_terminal(self) -> bool:
        return self.check_winner() is not None or len(self.get_legal_actions()) == 0

    def get_reward(self) -> float:
        winner = self.check_winner()

        match winner:
            case None:
                return 0.0
            case self.initial_player:
                return 1.0
            case _:
                return -1.0

    def check_winner(self) -> typing.Optional[typing.Any]:
        # check rows for a win
        for row in range(3):
            if self.board[row][0] is not None and self.board[row][0] == self.board[row][1] == self.board[row][2]:
                return self.board[row][0]

        # check columns for a win
        for col in range(3):
            if self.board[0][col] is not None and self.board[0][col] == self.board[1][col] == self.board[2][col]:
                return self.board[0][col]

        # check diagonals for a win
        if self.board[0][0] is not None and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            return self.board[0][0]
        if self.board[0][2] is not None and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            return self.board[0][2]

        return None

    def next_player(self) -> typing.Any:
        return "O" if self.current_player == "X" else "X"

    def deep_copy(self) -> sandbox_rl.core.interfaces.IGameState:
        return TicTacToe(
            board=[row[:] for row in self.board],
            initial_player=self.initial_player,
            current_player=self.current_player
        )

    def __str__(self) -> str:
        # return a string representation of the board
        def cell_str(cell: typing.Optional[str]) -> str:
            return cell if cell is not None else " "

        rows: typing.List[str] = ["|".join(cell_str(cell) for cell in row) for row in self.board]

        return "\n-----\n".join(rows)
