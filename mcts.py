import numpy as np
import math
import collections
from typing import Optional, List, Tuple

from rubiks_cube import RubiksCube, RubiksAction
from adi import ADI


class RootNode:
    def __init__(self):
        """
        MCTS Root node, can be considered as the virtual parent of the main UCTNode
        """
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


class UCTNode:
    def __init__(self, game_state: np.ndarray, move: Optional[int], actions_number: int, parent=None) -> None:
        """
        Node object of the MCTS
        :param game_state: State of the rubik's cube associated to the node
        :param move: Last move that has led to the current state of the node
        :param actions_number: Number of actions possible
        :param parent: Parent UCTNode
        :return None
        """
        self.game_state = game_state
        self.move = move
        self.actions_number = actions_number
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros(actions_number, dtype=np.float32)
        self.child_total_value = np.zeros(actions_number, dtype=np.float32)
        self.child_number_visits = np.zeros(actions_number, dtype=np.float32)

    @property
    def number_visits(self) -> int:
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value: int) -> None:
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self) -> float:
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value: int) -> None:
        self.parent.child_total_value[self.move] = value

    def child_Q(self) -> np.ndarray:
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self) -> np.ndarray:
        return math.sqrt(self.number_visits) * (self.child_priors /
                                                (1 + self.child_number_visits))

    def best_child(self) -> int:
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self) -> Tuple['UCTNode', List]:
        path = []
        current = self
        while current.is_expanded:
            current.number_visits += 1
            current.total_value -= 1
            best_action = current.best_child()
            path.append(best_action)
            current = current.maybe_add_child(best_action)
        return current, path

    def expand(self, child_priors: float) -> None:
        self.is_expanded = True
        self.child_priors = child_priors

    def maybe_add_child(self, move: int) -> 'UCTNode':
        if move not in self.children:
            rubiks_child = RubiksCube(cube=RubiksCube.from_one_hot_cube(self.game_state))
            state_child, _, _, _ = rubiks_child.step(RubiksAction(rubiks_child.actions[move]))
            self.children[move] = UCTNode(
                game_state=rubiks_child.to_one_hot_cube(state_child), move=move,
                actions_number=self.actions_number, parent=self
            )
        return self.children[move]

    def backup(self, value_estimate: float) -> None:
        current = self
        while current.parent is not None:
            current.total_value = max(value_estimate, current.total_value) + 1
            current = current.parent


class MCTS:
    def __init__(self, adi_model: ADI) -> None:
        """
        MCTS UCT algorithm implementation for Rubik's cubes resolution
        :param adi_model: Trained ADI model
        :return None
        """
        self.adi_model = adi_model
        self.main_node = None

    def search(self, rubiks, max_iterations) -> Optional[List]:
        """
        Actual resolution method for a given Rubik's cube object received
        :param rubiks: Rubiks's cube object to be resolved
        :param max_iterations: Number of tree explorations
        :return None
        """
        self.main_node = UCTNode(
            rubiks.state_one_hot, move=None,
            actions_number=len(rubiks.actions), parent=RootNode()
        )
        for _ in range(max_iterations):
            leaf, path = self.main_node.select_leaf()
            value, policy = self.adi_model.model.predict(np.expand_dims(leaf.game_state, axis=0))
            value, policy = np.asscalar(value), np.ravel(policy)
            leaf.expand(policy)
            leaf.backup(value)
            if RubiksCube(cube=leaf.game_state).is_resolved():
                return path
        return None

    def score(self, shuffle_depth: int, resolutions_number: int, max_iterations: int) -> float:
        """
        Scoring method of the MCTS, compute the number of cubes resolved according to a shuffle depth
        :param shuffle_depth: Number of shuffles of the cubes to be resolved
        :param resolutions_number: Number of cubes to be resolved
        :param max_iterations: Maximum iterations allowed for the MCTS algorithm
        :return score
        """
        score = 0
        for _ in range(resolutions_number):
            rubiks = RubiksCube(shuffle=False)
            rubiks.shuffle_cube(n=shuffle_depth)
            score += self.search(rubiks, max_iterations)
        return score
