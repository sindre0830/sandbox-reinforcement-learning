from typing import List, Optional, Tuple


class TicTacToe:
    def __init__(
        self,
        board: Optional[List[List[Optional[str]]]] = None,
        current_player: str = "X",
    ) -> None:
        # initialize the board and current player
        if board is None:
            self.board: List[List[Optional[str]]] = [[None for _ in range(3)] for _ in range(3)]
        else:
            self.board = board
        self.current_player: str = current_player

    def copy(self) -> "TicTacToe":
        new_board: List[List[Optional[str]]] = [row[:] for row in self.board]

        return TicTacToe(new_board, self.current_player)

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        # return a list of (row, col) tuples for empty cells
        actions: List[Tuple[int, int]] = []

        for i in range(3):
            for j in range(3):
                if self.board[i][j] is None:
                    actions.append((i, j))

        return actions

    def perform_action(self, action: Tuple[int, int]) -> "TicTacToe":
        # return a new state after applying the given action
        new_state: TicTacToe = self.copy()
        row, col = action
        if new_state.board[row][col] is not None:
            raise ValueError("cell is already occupied")

        new_state.board[row][col] = new_state.current_player
        new_state.current_player = "O" if new_state.current_player == "X" else "X"

        return new_state

    def is_terminal(self) -> bool:
        return self.check_winner() is not None or len(self.get_legal_actions()) == 0

    def check_winner(self) -> Optional[str]:
        # check rows for a win
        for i in range(3):
            if self.board[i][0] is not None and self.board[i][0] == self.board[i][1] == self.board[i][2]:
                return self.board[i][0]

        # check columns for a win
        for j in range(3):
            if self.board[0][j] is not None and self.board[0][j] == self.board[1][j] == self.board[2][j]:
                return self.board[0][j]

        # check diagonals for a win
        if self.board[0][0] is not None and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            return self.board[0][0]
        if self.board[0][2] is not None and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            return self.board[0][2]

        return None

    def get_reward(self) -> float:
        winner: Optional[str] = self.check_winner()

        if winner == "X":
            return 1.0
        elif winner == "O":
            return -1.0
        else:
            return 0.0

    def __str__(self) -> str:
        # return a string representation of the board
        def cell_str(cell: Optional[str]) -> str:
            return cell if cell is not None else " "

        rows: List[str] = ["|".join(cell_str(cell) for cell in row) for row in self.board]

        return "\n-----\n".join(rows)
