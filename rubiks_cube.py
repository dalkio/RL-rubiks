import numpy as np
import pandas as pd
import random
import collections
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rubiks_cube_config as rc_conf

class RubiksAction:
    def __init__(self, action=None):
        self.sides = rc_conf.sides
        self.directions = rc_conf.directions
        self.action = self._load_action(action) if action is not None else self._random_action()
            
    def _load_action(self, action):
        if (isinstance(action, str) and len(action) == 2 and
        action[0] in self.sides and action[1] in self.directions):
            return collections.namedtuple('Action', 'side direction')(*tuple(action)) 
        else:
            print('Unknown action!')
            return None

    def _random_action(self):
        action = random.choice(self.sides) + random.choice(self.directions)
        return collections.namedtuple('Action', 'side direction')(*tuple(action)) 

class RubiksCube:
    def __init__(self, cube=None, verbose=True, shuffle=True):
        self.dim = rc_conf.dim
        self.colors = rc_conf.colors
        self.connexions = rc_conf.connexions
        self.sides = rc_conf.sides
        self.directions = rc_conf.directions
        self.actions = [side+direction for side, direction in product(self.sides, self.directions)]
        self.index_colors = {color: index for index, color in enumerate(self.colors)}
        self.index_sides = {side: index for index, side in enumerate(self.sides)}
        self.index_actions = {action: index for index, action in enumerate(self.actions)}
        if cube is None:
            self.cube = self._construct_cube()
        else:
            if isinstance(cube, np.ndarray) and cube.shape == (len(self.colors), self.dim, self.dim):
                self.cube = cube.copy()
            else:
                print('Incorrect cube format!')
        self.counter = 0
        self.verbose = verbose
        if shuffle and cube is None:
            self._shuffle_cube()

    def _construct_cube(self, verbose=True):
        cube = np.empty((len(self.colors), self.dim, self.dim), dtype='int64')
        for index in self.index_colors.values():
            cube[index, :, :] = index
        if verbose:
            print('Cube initialized!')
        return cube

    def _rotate_helper(self, matrix, direction):
        if direction == 'd':
            return np.rot90(matrix, k=1)
        elif direction == 'i':
            return np.rot90(matrix, k=3)
        
    def _edge_translation(self, side, side_matrix_a, side_matrix_b, edge_a, edge_b, 
                          return_array=True, input_array=None):
        def _edge_to_slice(edge):
            if edge == 'r':
                return np.s_[:, -1]
            elif edge == 'l':
                return np.s_[:, 0]
            elif edge == 'u':
                return np.s_[0, :]
            elif edge == 'd':
                return np.s_[-1, :]
            else:
                print('Invalid edge!')
                return None
        if input_array is None:
            edge_array_a = side_matrix_a[_edge_to_slice(edge_a)]
        else:
            edge_array_a = input_array
        if (edge_a, edge_b) in self.connexions[side]['inversions']:
            edge_array_a = edge_array_a[::-1]
        edge_array_b = side_matrix_b[_edge_to_slice(edge_b)].copy()
        side_matrix_b[_edge_to_slice(edge_b)] = edge_array_a
        return edge_array_b if return_array else None
                
    def _rotate(self, action, verbose=True):
        side, direction = action.side, action.direction
        side_matrix = self.cube[self.index_sides[side]]
        connected_sides = self.connexions[side]['sides']
        connected_side_matrices = [self.cube[self.index_sides[connected_side]] 
                                   for connected_side in connected_sides]
        connected_edges = self.connexions[side]['edges']
        if direction == 'i':
            connected_sides, connected_side_matrices, connected_edges = \
                connected_sides[::-1], connected_side_matrices[::-1], connected_edges[::-1]
        self.cube[self.index_sides[side]] = self._rotate_helper(side_matrix, direction)
        previous_array = None
        for idx in range(len(connected_side_matrices)-1):
            previous_array = self._edge_translation(
                side,
                connected_side_matrices[idx], connected_side_matrices[idx+1],
                connected_edges[idx], connected_edges[idx+1],
                input_array=previous_array
            )
        self.counter += 1
        if verbose:
            print('{0}: {1} done, reward={2}'.format(str(self.counter), action, self._get_reward()))

    def _shuffle_cube(self, n=100):
        for i in range(n):
            random_action = RubiksAction()
            self._rotate(random_action.action, verbose=False)
            self.counter = 0
        if self.verbose:
            print('Cube shuffled {0} times'.format(n))
            
    def _get_reward(self):
        return np.sum([np.sum(self.cube[i]==i) for i in self.index_colors.values()])

    def is_resolved(self):
        return np.all([np.all(self.cube[i]==i) for i in self.index_colors.values()])
    
    def reset(self, shuffle=True):
        self.cube = self._construct_cube()
        if shuffle:
            self._shuffle_cube()
            
    def step(self, action):
        assert isinstance(action, RubiksAction) and action.action is not None
        self._rotate(action.action, verbose=self.verbose)
        reward = self._get_reward()
        done = self.is_resolved()
        return reward, done
    
    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        r = list(range(self.dim+1))
        X, Y = np.meshgrid(r, r[::-1])
        zero, three = np.zeros(4).reshape(2, 2), np.ones(4).reshape(2, 2) * 3
        edge_color, edge_width = 'black', 3
        for x_idx in range(X.shape[0]-1):
            for y_idx in range(X.shape[1]-1):
                x = X[x_idx:x_idx+2, y_idx:y_idx+2]
                y = Y[x_idx:x_idx+2, y_idx:y_idx+2]
                ax.plot_wireframe(three, x, y, linewidth=edge_width, color=edge_color)
                ax.plot_surface(
                    three, x, y, color=self.colors[self.cube[0][x_idx][y_idx]]
                )
                ax.plot_wireframe(zero, x, y, linewidth=edge_width, color=edge_color)
                ax.plot_surface(
                    zero, x, y, color=self.colors[np.flip(self.cube[1], axis=1)[x_idx][y_idx]]
                )
                ax.plot_wireframe(x, y, three, linewidth=edge_width, color=edge_color)
                ax.plot_surface(
                    x, y, three, color=self.colors[self.cube[2][x_idx][y_idx]]
                )
                ax.plot_wireframe(x, y, zero, linewidth=edge_width, color=edge_color)
                ax.plot_surface(
                    x, y, zero, color=self.colors[np.flip(self.cube[3], axis=0)[x_idx][y_idx]]
                )
                ax.plot_wireframe(x, zero, y, linewidth=edge_width, color=edge_color)
                ax.plot_surface(
                    x, zero, y, color=self.colors[self.cube[4][x_idx][y_idx]]
                )
                ax.plot_wireframe(x, three, y, linewidth=edge_width, color=edge_color)
                ax.plot_surface(
                    x, three, y, color=self.colors[np.flip(self.cube[5], axis=1)[x_idx][y_idx]]
                )
        ax._axis3don = False
        plt.show()