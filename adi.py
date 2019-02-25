import logging
import sys
import os
import re
import numpy as np
from typing import Tuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from rubiks_cube import RubiksCube, RubiksAction


class ADI(object):
    def __init__(self, k: int = 25, l: int = 40000, load_files: Tuple[str, str] = None,
                 save_dataset: bool = True, save_model: bool = True,
                 verbose: bool = True, shuffle: bool = True) -> None:
        """
        Autodidactic Iteration Algorithm for Rubik's Cube solving using reinforcement learning
        https://arxiv.org/pdf/1805.07470.pdf
        :param k: Number of scrambles from the solved state to generate a sequence of cubes
        :param l: Number of sequences generated
        :param load_files: Tuple of dataset and weights filenames
        :param save_dataset: Boolean for saving or not the created dataset
        :param save_model: Boolean for saving or not the trained model
        :param verbose: Verbosity parameter
        :param shuffle: Dataset shuffle parameter
        """
        self.k = k
        self.l = l
        self.load_files = load_files
        self.verbose = verbose
        self.shuffle = shuffle
        self.X = None
        self.weights = None

        self.logger = self._create_logger()
        if self.load_files:
            self.k, self.l, self.X, self.weights = self._load_dataset()
        else:
            self.X, self.weights = self._generate_dataset()
        self.model = self._design_model()

    @staticmethod
    def _create_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(asctime)s — %(funcName)s() — %(levelname)s — %(message)s")
        if logger.handlers:
            logger.addHandler(logger.handlers[0])
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def _load_dataset(self) -> Tuple[int, int, np.ndarray, np.ndarray]:
        self.logger.info("Loading dataset...")
        assert len(self.load_files) == 2 and isinstance(self.load_files[0], str) \
            and isinstance(self.load_files[1], str), \
            "Bad format for load_files parameter"
        numbers_parsed = re.findall(r'\d+', self.load_files[0])
        assert len(numbers_parsed) == 2, \
            "Dataset file name incorrect"
        k, l = [int(digit) for digit in numbers_parsed]
        self.logger.info("k={0}, l={1}".format(k, l))
        X = np.load('data' + os.path.sep + self.load_files[0])
        weights = np.load('data' + os.path.sep + self.load_files[1])
        return k, l, X, weights

    def _generate_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        samples_file = "data/scrambled_cubes_k{0}_l{1}.npy".format(self.k, self.l)
        weights_file = "data/weights_k{0}_l{1}.npy".format(self.k, self.l)
        self.logger.info("Generating dataset...")
        X, weights = [], []
        for iteration in range(self.l):
            rubiks_cube = RubiksCube(shuffle=False, verbose=False)
            for shuffle in range(self.k):
                rubiks_cube.shuffle_cube(n=1)
                weight = 1 / (shuffle + 1)
                X.append(rubiks_cube.state_one_hot)
                weights.append(weight)
            if self.verbose and iteration % (self.l / 10) == 0:
                self.logger.info("{0:.2f}%".format((iteration / self.l) * 100))
        X, weights = np.asarray(X), np.asarray(weights)
        if self.shuffle:
            random_indexes = np.random.permutation(range(self.l * self.k))
            X, weights = X[random_indexes], weights[random_indexes]
        os.makedirs('data', exist_ok=True)
        np.save(samples_file, X)
        np.save(weights_file, weights)
        return X, weights

    @staticmethod
    def _design_model() -> Model:
        rubiks_cube = RubiksCube()

        inputs = Input(shape=rubiks_cube.state_one_hot.shape, name='input')
        x = Flatten()(inputs)
        x = Dense(4096, activation='elu')(x)
        x = Dense(2048, activation='elu')(x)
        pre_v = Dense(512, activation='elu')(x)
        pre_p = Dense(512, activation='elu')(x)
        v = Dense(1, activation='linear', name='value_output')(pre_v)
        p = Dense(12, activation='softmax', name='policy_output')(pre_p)

        losses = {
            'value_output': 'mean_squared_error',
            'policy_output': 'categorical_crossentropy'
        }

        model = Model(inputs=inputs, outputs=[v, p])
        model.compile(optimizer='rmsprop', loss=losses)
        return model
    
    def save_trained_model(self, filename) -> None:
        model_name_json, model_name_h5 = filename + '.json', filename + '.h5'
        model_json = self.model.to_json()
        with open(model_name_json, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_name_h5)

    def train(self, batch_size: int = 1000, batches_number: int = 5, epochs_per_batch: int = 1,
              save_frequency: int = 10) -> None:
        """
        :param batch_size: Number of cubes for a batch for one pass
        :param batches_number: Number of total epochs
        :param epochs_per_batch: Number of epochs for one batch
        :param save_frequency: Number of batches between each saving
        """
        self.logger.info("Training model...")
        rubiks_cube = RubiksCube()
        for batch_number in range(batches_number):
            self.logger.info("Batch number: {0}".format(batch_number+1))
            Y_p, Y_v = [], []
            batch_indexes = np.random.choice(range(len(self.X)), size=batch_size, replace=False)
            X_batch, weights_batch = self.X[batch_indexes], self.weights[batch_indexes]
            for iteration, X_i in enumerate(X_batch):
                rewards_i, values_i = [], []
                for action in rubiks_cube.actions:
                    rubiks_cube_copy = RubiksCube(cube=X_i)
                    _, reward_a, _, _ = rubiks_cube_copy.step(RubiksAction(action))
                    (v_x_i_a, p_x_i_a) = self.model.predict(np.expand_dims(rubiks_cube_copy.state_one_hot, axis=0))
                    rewards_i.append(reward_a)
                    values_i.append(np.asscalar(v_x_i_a))
                y_p_i = np.ravel(np.eye(1, len(rubiks_cube.actions), np.argmax(np.sum([rewards_i, values_i], axis=0))))
                y_v_i = np.atleast_1d(np.max(np.sum([rewards_i, values_i], axis=0)))
                Y_p.append(y_p_i)
                Y_v.append(y_v_i)
            Y_p, Y_v = np.asarray(Y_p), np.asarray(Y_v)
            self.model.fit(
                {'input': X_batch},
                {'policy_output': Y_p, 'value_output': Y_v},
                sample_weight={'policy_output': weights_batch, 'value_output': weights_batch},
                epochs=epochs_per_batch
            )
            if batch_number%save_frequency == 0 and batch_number != 0:
                filename = "data/model_k{0}_l{1}_iter{2}".format(self.k, self.l, batch_number)
                self.save_trained_model(filename)