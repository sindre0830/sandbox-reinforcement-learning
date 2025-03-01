import pytest
import sandbox_rl.application.game_states


def test_get_legal_actions_empty_board():
    # arrange
    game_state = sandbox_rl.application.game_states.TicTacToe()
    expected_actions = [(i, j) for i in range(3) for j in range(3)]

    # act
    actions = game_state.get_legal_actions()

    # assert
    assert set(actions) == set(expected_actions)


def test_get_legal_actions_partially_filled_board():
    # arrange
    board = [
        ["X", None, "O"],
        [None, "X", None],
        [None, None, "O"]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_actions = [(0, 1), (1, 0), (1, 2), (2, 0), (2, 1)]

    # act
    actions = game_state.get_legal_actions()

    # assert
    assert set(actions) == set(expected_actions)


def test_get_legal_actions_full_board_returns_empty_list():
    # arrange
    board = [
        ["X", "O", "X"],
        ["O", "X", "O"],
        ["X", "O", "X"]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_actions = []

    # act
    actions = game_state.get_legal_actions()

    # assert
    assert actions == expected_actions


def test_get_legal_actions_single_empty_cell():
    # arrange
    board = [
        ["X", "O", "X"],
        ["O", None, "O"],
        ["X", "O", "X"]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_actions = [(1, 1)]

    # act
    actions = game_state.get_legal_actions()

    # assert
    assert actions == expected_actions


def test_perform_action_updates_board_and_switches_player():
    # arrange
    game_state = sandbox_rl.application.game_states.TicTacToe()
    initial_player = game_state.current_player
    action = (0, 0)

    # act
    new_game_state = game_state.perform_action(action)

    # assert
    # check that the move is recorded on the board
    assert new_game_state.board[0][0] == initial_player
    # check that the current player is switched
    assert new_game_state.current_player != initial_player
    # check that the original board remains unchanged
    assert game_state.board[0][0] is None


def test_perform_action_invalid():
    # arrange
    game_state = sandbox_rl.application.game_states.TicTacToe()
    action = (1, 1)
    game_state = game_state.perform_action(action)

    # act & assert
    with pytest.raises(ValueError):
        game_state.perform_action(action)


def test_is_terminal_with_win():
    # arrange
    board = [
        ["X", "X", "X"],
        [None, "O", None],
        ["O", None, None]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board, initial_player="X", current_player="O"
    )

    # act
    terminal = game_state.is_terminal()
    winner = game_state.check_winner()

    # assert
    assert terminal is True
    assert winner == "X"


def test_is_terminal_with_tie():
    # arrange
    board = [
        ["X", "O", "X"],
        ["X", "O", "O"],
        ["O", "X", "X"]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board, initial_player="X", current_player="O"
    )

    # act
    terminal = game_state.is_terminal()
    winner = game_state.check_winner()

    # assert
    assert terminal is True
    assert winner is None


def test_get_reward_win_initial_player():
    # arrange
    board = [
        ["X", "X", "X"],
        [None, "O", None],
        ["O", None, None]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board, initial_player="X", current_player="O"
    )

    # act
    reward = game_state.get_reward()

    # assert
    assert reward == 1.0


def test_get_reward_win_opponent():
    # arrange
    board = [
        ["O", "O", "O"],
        [None, "X", None],
        ["X", None, None]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board, initial_player="X", current_player="O"
    )

    # act
    reward = game_state.get_reward()

    # assert
    assert reward == -1.0


def test_get_reward_no_winner():
    # arrange
    board = [
        ["X", "O", "X"],
        ["X", "O", "O"],
        ["O", "X", "X"]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board, initial_player="X", current_player="O"
    )

    # act
    reward = game_state.get_reward()

    # assert
    assert reward == 0.0


def test_next_player():
    # arrange
    game_state = sandbox_rl.application.game_states.TicTacToe(current_player="X")

    # act
    next_player = game_state.next_player()

    # assert
    assert next_player == "O"
    # arrange again
    game_state.current_player = "O"
    # act
    next_player = game_state.next_player()
    # assert
    assert next_player == "X"


def test_deep_copy_independence():
    # arrange
    game_state = sandbox_rl.application.game_states.TicTacToe()

    # act
    copied_state = game_state.deep_copy()
    copied_state.board[0][0] = "X"

    # assert
    assert game_state.board[0][0] is None


def test_str_representation():
    # arrange
    board = [
        ["X", None, "O"],
        [None, "X", None],
        ["O", None, "X"]
    ]
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_str = "X| |O\n-----\n |X| \n-----\nO| |X"

    # act
    board_str = str(game_state)

    # assert
    assert board_str == expected_str
