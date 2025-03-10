import math
import sandbox_rl.application.learning_agents


def test_node_is_leaf():
    # assign
    node = sandbox_rl.application.learning_agents.MCTS.Node(state=object())
    # act
    result = node.is_leaf()
    # assert
    assert result is True


def test_node_update():
    # assign
    node = sandbox_rl.application.learning_agents.MCTS.Node(state=object())
    # act
    node.update(1.0)
    # assert
    assert node.visit_count == 1
    assert node.total_value == 1.0


def test_node_backpropagation():
    # assign
    parent = sandbox_rl.application.learning_agents.MCTS.Node(state=object())
    child = sandbox_rl.application.learning_agents.MCTS.Node(state=object(), parent=parent)
    # act
    child.backpropagate(1.0)
    # assert
    assert child.visit_count == 1
    assert child.total_value == 1.0
    assert parent.visit_count == 1
    assert parent.total_value == -1.0


def test_node_puct_score():
    # assign
    parent = sandbox_rl.application.learning_agents.MCTS.Node(state=object())
    child = sandbox_rl.application.learning_agents.MCTS.Node(
        state=object(),
        parent=parent,
        prior_probability=0.5
    )
    parent.visit_count = 9
    child.visit_count = 3
    child.total_value = 2.0
    c_puct = 1.4
    expected_U = c_puct * 0.5 * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    expected_Q = child.total_value / child.visit_count
    expected_score = expected_Q + expected_U
    # act
    score = child.puct_score(c_puct)
    # assert
    assert abs(score - expected_score) < 1e-5


def test_get_action_probabilities_nonzero_temperature():
    # assign
    mcts = sandbox_rl.application.learning_agents.MCTS(
        initial_game_state=object(),
        game_agent=None,
        replay_buffer=None,
        episodes=1,
        simulations=1,
        train_every=1,
        temperature=1.0,
    )
    root = sandbox_rl.application.learning_agents.MCTS.Node(state=object())
    child1 = sandbox_rl.application.learning_agents.MCTS.Node(state=object(), parent=root)
    child1.visit_count = 4
    child2 = sandbox_rl.application.learning_agents.MCTS.Node(state=object(), parent=root)
    child2.visit_count = 1
    root.children = {(0, 0): child1, (0, 1): child2}
    # act
    action_probs = mcts.get_action_probabilities(root)
    # assert
    expected_probs = {(0, 0): 4 / 5, (0, 1): 1 / 5}
    for action, prob in action_probs.items():
        assert abs(prob - expected_probs[action]) < 1e-5
