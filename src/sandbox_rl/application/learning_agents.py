import sandbox_rl.core.interfaces
import sandbox_rl.core.models
import interface
import typing
import numpy as np
import copy


class MCTS(interface.implements(sandbox_rl.core.interfaces.ILearningAgent)):
    def __init__(
        self,
        initial_game_state: sandbox_rl.core.interfaces.IGameState,
        game_agent: sandbox_rl.core.interfaces.IGameAgent,
        replay_buffer: sandbox_rl.core.interfaces.IReplayBuffer,
        episodes: int,
        simulations: int,
        batch_size: int = 32,
        train_every: int = 1,
        c_puct: float = 1.4,
        temperature: float = 1.0,
    ) -> None:
        self.initial_game_state = initial_game_state
        self.game_agent = game_agent
        self.replay_buffer = replay_buffer
        self.episodes = episodes
        self.simulations = simulations
        self.train_every = train_every
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.temperature = temperature

    def execute(self) -> None:
        for episode in range(1, self.episodes + 1):
            self.self_play()

            if episode % self.train_every == 0:
                batch = self.replay_buffer.sample(self.batch_size)
                self.game_agent.train(batch)

    def self_play(self) -> None:
        state = copy.deepcopy(self.initial_game_state)
        episode_data = []

        while not state.is_terminal():
            root = self.search(state)
            action_probs = self.get_action_probabilities(root)

            episode_data.append((state, action_probs))

            actions, probabilities = zip(*action_probs.items())
            action = actions[np.random.choice(len(actions), p=probabilities)]
            state = state.perform_action(action)

        final_value = state.get_reward()

        for state, policy in episode_data:
            self.replay_buffer.store((state.encode_state(), np.array(list(policy.values())), final_value))
            final_value = -final_value

    def search(self, initial_state: sandbox_rl.core.interfaces.IGameState) -> "MCTS.Node":
        root = MCTS.Node(state=initial_state)

        for _ in range(self.simulations):
            node = root

            # selection
            while not node.is_leaf():
                node = node.select(self.c_puct)

            # expansion and evaluation
            if not node.state.is_terminal():
                actions = node.state.get_legal_actions()
                prior_probs, value = self.game_agent.select_action(node.state)
                node.expand(actions, prior_probs)
            else:
                value = node.state.get_reward()

            node.backpropagate(value)

        return root

    def get_action_probabilities(self, root: "MCTS.Node") -> typing.Dict[typing.Tuple[int, int], float]:
        action_visits = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())

        if self.temperature == 0:
            best_actions = np.zeros_like(action_visits)
            best_actions = np.argwhere(action_visits == np.max(action_visits)).flatten()
            action_probs = np.zeros_like(action_visits)
            action_probs[np.random.choice(best_actions)] = 1.0
        else:
            action_visits = action_visits ** (1 / self.temperature)
            action_probs = action_visits / np.sum(action_visits)

        return dict(zip(actions, action_probs))

    class Node():
        def __init__(
            self,
            state: sandbox_rl.core.interfaces.IGameState,
            parent: "MCTS.Node" = None,
            prior_probability: float = 0.0,
        ) -> None:
            self.state: sandbox_rl.core.interfaces.IGameState = state
            self.parent: MCTS.Node = parent
            self.children: typing.Dict[typing.Any, MCTS.Node] = {}
            self.visit_count: int = 0
            self.total_value: float = 0.0
            self.prior_probability: float = prior_probability

        def is_leaf(self) -> bool:
            return len(self.children) == 0

        def expand(self, actions: np.ndarray, prior_probabilities: np.ndarray) -> None:
            for action, prior_probability in zip(actions, prior_probabilities):
                new_state = self.state.perform_action(action)

                self.children[tuple(action)] = MCTS.Node(
                    state=new_state,
                    parent=self,
                    prior_probability=prior_probability,
                )

        def select(self, c_puct: float) -> "MCTS.Node":
            return max(self.children.values(), key=lambda child: child.puct_score(c_puct))

        def puct_score(self, c_puct: float) -> float:
            U = c_puct * self.prior_probability * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
            Q = (self.total_value / self.visit_count) if self.visit_count > 0 else 0.0

            return Q + U

        def update(self, value: float) -> None:
            self.visit_count += 1
            self.total_value += value

        def backpropagate(self, value: float) -> None:
            self.update(value)

            if self.parent is not None:
                self.parent.backpropagate(-value)
