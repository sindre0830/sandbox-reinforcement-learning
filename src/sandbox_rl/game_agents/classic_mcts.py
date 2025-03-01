import math
import random
from typing import Dict, Generic, List, Optional, TypeVar

State = TypeVar("State")
Action = TypeVar("Action")


class ClassicMctsNode(Generic[State, Action]):
    def __init__(self, state: State, parent: Optional["ClassicMctsNode[State, Action]"] = None) -> None:
        self.state: State = state
        self.parent: Optional["ClassicMctsNode[State, Action]"] = parent
        self.children: Dict[Action, "ClassicMctsNode[State, Action]"] = {}
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.untried_actions: Optional[List[Action]] = None

    def is_fully_expanded(self) -> bool:
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, exploration_constant: float) -> Optional["ClassicMctsNode[State, Action]"]:
        best_value: float = -float("inf")
        best_node: Optional["ClassicMctsNode[State, Action]"] = None

        for _, child in self.children.items():
            if child.visits == 0:
                continue

            uct_value: float = (child.total_reward / child.visits) + exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits
            )

            if uct_value > best_value:
                best_value = uct_value
                best_node = child

        return best_node


class ClassicMcts(Generic[State, Action]):
    def __init__(
        self,
        root_state: State,
        exploration_constant: float = math.sqrt(2),
        max_iterations: int = 1000,
    ) -> None:
        self.root: ClassicMctsNode[State, Action] = ClassicMctsNode(root_state)
        self.exploration_constant: float = exploration_constant
        self.max_iterations: int = max_iterations

    def search(self) -> Action:
        for _ in range(self.max_iterations):
            node: ClassicMctsNode[State, Action] = self.selection(self.root)
            reward: float = self.simulation(node.state)
            self.backpropagation(node, reward)

        return self.best_action()

    def selection(self, node: ClassicMctsNode[State, Action]) -> ClassicMctsNode[State, Action]:
        while not self.is_terminal(node.state):
            if node.untried_actions is None:
                node.untried_actions = self.get_legal_actions(node.state)

            if node.untried_actions:
                return self.expansion(node)

            next_node: Optional[ClassicMctsNode[State, Action]] = node.best_child(self.exploration_constant)
            if next_node is None:
                break

            node = next_node

        return node

    def expansion(self, node: ClassicMctsNode[State, Action]) -> ClassicMctsNode[State, Action]:
        action: Action = node.untried_actions.pop()
        next_state: State = self.perform_action(node.state, action)

        child_node: ClassicMctsNode[State, Action] = ClassicMctsNode(next_state, parent=node)
        child_node.untried_actions = self.get_legal_actions(next_state)

        node.children[action] = child_node

        return child_node

    def simulation(self, state: State) -> float:
        current_state: State = state
        while not self.is_terminal(current_state):
            legal_actions: List[Action] = self.get_legal_actions(current_state)
            if not legal_actions:
                break

            action: Action = random.choice(legal_actions)
            current_state = self.perform_action(current_state, action)

        return self.get_reward(current_state)

    def backpropagation(self, node: ClassicMctsNode[State, Action], reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def best_action(self) -> Action:
        best_visits: int = -1
        best_action: Optional[Action] = None

        for action, child in self.root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action

        if best_action is None:
            raise RuntimeError("No best action found.")

        return best_action

    def is_terminal(self, state: State) -> bool:
        """Return True if the state is terminal. Override this method."""
        raise NotImplementedError("is_terminal must be implemented by the domain.")

    def get_legal_actions(self, state: State) -> List[Action]:
        """Return a list of legal actions from the given state. Override this method."""
        raise NotImplementedError("get_legal_actions must be implemented by the domain.")

    def perform_action(self, state: State, action: Action) -> State:
        """Return the next state after performing the given action. Override this method."""
        raise NotImplementedError("perform_action must be implemented by the domain.")

    def get_reward(self, state: State) -> float:
        """Return the reward for the terminal state. Override this method."""
        raise NotImplementedError("get_reward must be implemented by the domain.")
