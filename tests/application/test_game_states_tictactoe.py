import pytest
import sandbox_rl.application.game_states
import sandbox_rl.core.constants
import numpy as np


def test_get_legal_actions_empty_board():
    # arrange
    game_state = sandbox_rl.application.game_states.TicTacToe()
    expected_actions = np.array([(i, j) for i in range(3) for j in range(3)])

    # act
    actions = game_state.get_legal_actions()

    # assert
    assert actions.tobytes() == expected_actions.tobytes()


def test_get_legal_actions_partially_filled_board():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2],
        [sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY],
        [sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_actions = np.array([(0, 1), (1, 0), (1, 2), (2, 0), (2, 1)])

    # act
    actions = game_state.get_legal_actions()

    # assert
    assert actions.tobytes() == expected_actions.tobytes()


def test_get_legal_actions_full_board_returns_empty_list():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1],
        [sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2],
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_actions = np.array([])

    # act
    actions = game_state.get_legal_actions()

    # assert
    assert actions.tobytes() == expected_actions.tobytes()


def test_get_legal_actions_single_empty_cell():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1],
        [sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2],
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_actions = np.array([(1, 1)])

    # act
    actions = game_state.get_legal_actions()

    # assert
    assert actions.tobytes() == expected_actions.tobytes()


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
    assert game_state.board[0][0] == sandbox_rl.core.constants.EMPTY


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
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_1],
        [sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.EMPTY],
        [sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.EMPTY]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board,
        initial_player=sandbox_rl.core.constants.PLAYER_1,
        current_player=sandbox_rl.core.constants.PLAYER_2
    )

    # act
    terminal = game_state.is_terminal()
    winner = game_state.check_winner()

    # assert
    assert terminal is True
    assert winner == sandbox_rl.core.constants.PLAYER_1


def test_is_terminal_with_tie():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1],
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_2],
        [sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_1]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board,
        initial_player=sandbox_rl.core.constants.PLAYER_1,
        current_player=sandbox_rl.core.constants.PLAYER_2
    )

    # act
    terminal = game_state.is_terminal()
    winner = game_state.check_winner()

    # assert
    assert terminal is True
    assert winner == sandbox_rl.core.constants.TIE


def test_get_reward_win_initial_player():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_1],
        [sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.EMPTY],
        [sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.EMPTY]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board,
        initial_player=sandbox_rl.core.constants.PLAYER_1,
        current_player=sandbox_rl.core.constants.PLAYER_2
    )

    # act
    reward = game_state.get_reward()

    # assert
    assert reward == 1.0


def test_get_reward_win_opponent():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_2],
        [sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY],
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.EMPTY]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board,
        initial_player=sandbox_rl.core.constants.PLAYER_1,
        current_player=sandbox_rl.core.constants.PLAYER_2
    )

    # act
    reward = game_state.get_reward()

    # assert
    assert reward == -1.0


def test_get_reward_no_winner():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1],
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_2],
        [sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.PLAYER_1]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(
        board=board,
        initial_player=sandbox_rl.core.constants.PLAYER_1,
        current_player=sandbox_rl.core.constants.PLAYER_2
    )

    # act
    reward = game_state.get_reward()

    # assert
    assert reward == 0.0


def test_next_player():
    # arrange
    game_state = sandbox_rl.application.game_states.TicTacToe(
        current_player=sandbox_rl.core.constants.PLAYER_1
    )

    # act
    next_player = game_state.next_player()

    # assert
    assert next_player == sandbox_rl.core.constants.PLAYER_2
    # arrange again
    game_state.current_player = sandbox_rl.core.constants.PLAYER_2
    # act
    next_player = game_state.next_player()
    # assert
    assert next_player == sandbox_rl.core.constants.PLAYER_1


def test_str_representation():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2],
        [sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY],
        [sandbox_rl.core.constants.PLAYER_2, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_1]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_str = "X| |O\n-----\n |X| \n-----\nO| |X"

    # act
    board_str = str(game_state)

    # assert
    assert board_str == expected_str


def test_encode_state():
    # arrange
    board = np.array([
        [sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2],
        [sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY],
        [sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2]
    ])
    game_state = sandbox_rl.application.game_states.TicTacToe(board=board)
    expected_array = np.array([
        sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2,
        sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_1, sandbox_rl.core.constants.EMPTY,
        sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.EMPTY, sandbox_rl.core.constants.PLAYER_2
    ])

    # act
    array = game_state.encode_state()

    # assert
    assert array.tobytes() == expected_array.tobytes()
