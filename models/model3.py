import json
from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from lib.connectfour import Game
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from tensorflow import keras
from tensorflow.python.keras.models import load_model

from models.model2 import DRAW

from multiprocessing import Pool

LOSS = "categorical_crossentropy"

TRAIN_SIZE = 0.7
ITERATIONS = 100
MC_DEPTH = 10

EXTRA_FEATURES = 1
NUM_ROWS = 6
NUM_COLS = 7

INPUT_SIZE = (NUM_COLS * NUM_ROWS) + EXTRA_FEATURES
OUTPUT_SIZE = 3

PLAYER_RANDOM = -1
PLAYER_AI = 1
DRAWS = 0

LEARNING_RATE = 1e-3
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
LAYERS = [300, 225, 225, 225]

ACTIVATION_OUTPUT = "softmax"
ACTIVATION_HIDDEN = "relu"

BATCH_SIZE = 32
EPOCHS = 25


class Model3:
    _model: Sequential
    _dataset = []

    def __init__(self, dataset=[]) -> None:
        self._dataset = dataset

    def get_input_output(self):
        input, output = [], []

        for data in self._dataset:
            board, results, features = data[0], data[1], data[2:]

            input.append(np.append(board, features))
            output.append(results)

        X = np.array(input).reshape((-1, INPUT_SIZE))
        y = to_categorical(output, OUTPUT_SIZE)

        return X, y

    def get_split_input_output(self):
        X, y = self.get_input_output()
        return train_test_split(X, y, train_size=TRAIN_SIZE)

    def create_model(self):
        self._model = Sequential()

        # Add input layer
        self._model.add(Dropout(0.2, input_shape=(INPUT_SIZE,)))

        # Add hidden Layers
        for layer in LAYERS:
            self._model.add(Dense(layer, activation=ACTIVATION_HIDDEN))

        # Add output Layer
        self._model.add(Dense(OUTPUT_SIZE, activation=ACTIVATION_OUTPUT))

        self._model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])

        return self._model

    def train_model(self, batch_size, name):
        X_train, X_test, y_train, y_test = self.get_split_input_output()

        history = self._model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=batch_size,
        )

        self._model.save("data/models/" + name)

        return history

    def train_model_xy(self, X, y):
        return self._model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    def load_model(self, name: str):
        self._model = load_model("data/models/" + name)

    def predict(self, board, features, id):
        input = np.array(np.append(board, features)).reshape((-1, INPUT_SIZE))
        return self._model.predict(input)[0][id]
        
    def predict_move(
        self, game: Game, player_id: int, starting_player: int, legal_only: bool = True, num_cols: int = 7,
    ):

        best_column, highest_prediction = 3, 0

        for idx in range(num_cols):
            if legal_only and not game.is_legal_move(idx):
                continue

            temp_game = deepcopy(game)
            temp_game.play_move(player_id, idx)

            pred_val = self.predict(board=temp_game.board, features=[starting_player], id=1 if player_id == 1 else 2)

            if highest_prediction <= pred_val:
                best_column = idx
                highest_prediction = pred_val

        return best_column

    def validate_against_random(self):
        iterations = 200
        result_values = {PLAYER_AI: "ai",
                         PLAYER_RANDOM: "random", DRAWS: "draw"}
        starts_values = {PLAYER_AI: "ai",
                         PLAYER_RANDOM: "random"}

        results = {PLAYER_AI: 0, PLAYER_RANDOM: 0, DRAWS: 0}
        starts = {PLAYER_AI: 0, PLAYER_RANDOM: 0}

        for i in range(iterations):

            game = Game()

            active_player = PLAYER_AI if i < (
                    iterations / 2) else PLAYER_RANDOM

            start_player = active_player

            while game.check_status() == None:
                move = 3

                if active_player == PLAYER_AI:
                    best_move = self.predict_move(game=game, player_id=active_player, starting_player=start_player)
                    move = best_move
                else:
                    random_move = game.random_action(legal_only=True)
                    move = random_move

                game.play_move(player=active_player, column=move)
                active_player *= -1

            starts[start_player] += 1
            results[game.status] += 1

            # print("iteration ({0}): win({1}) and starts({2})".format(
            #     i, result_values[game.status], starts_values[start_player]))

        results_ai = results[PLAYER_AI]
        results_random = results[PLAYER_RANDOM]
        results_draw = results[DRAWS]

        win_rate = results_ai / (results_ai + results_random)

        print("win-rate: {0}% and {1} draws".format(win_rate, results_draw))

        return results, starts

    def validate_against_monte_carlo(self, n=5):
        iterations = 100
        result_values = {PLAYER_AI: "ai", PLAYER_RANDOM: "random", DRAW: "draw"}
        starts_values = {PLAYER_AI: "ai", PLAYER_RANDOM: "random"}

        results = {PLAYER_AI: 0, PLAYER_RANDOM: 0, DRAW: 0}
        starts = {PLAYER_AI: 0, PLAYER_RANDOM: 0}

        for i in range(iterations):

            game = Game()

            active_player = PLAYER_AI if i < (iterations / 2) else PLAYER_RANDOM
            start_player = active_player

            while game.check_status() == None:
                move = 3

                if active_player == PLAYER_AI:
                    best_move = self.predict_move(game=game, player_id=active_player, starting_player=start_player)
                    move = best_move
                else:
                    move, _ = game.smart_action(player=active_player, legal_only=True, n=n)

                game.play_move(player=active_player, column=move)
                active_player *= -1

            starts[start_player] += 1
            results[game.status] += 1

            results_ai = results[PLAYER_AI]
            results_random = results[PLAYER_RANDOM]
            results_draw = results[DRAW]

            win_rate = results_ai / (results_ai + results_random)

            print("iteration ({0}): win({1}) and starts({2}) and win-rate ({3})".format(i, result_values[game.status],
                                                                                        starts_values[start_player],
                                                                                        win_rate))

        results_ai = results[PLAYER_AI]
        results_random = results[PLAYER_RANDOM]
        results_draw = results[DRAW]

        win_rate = results_ai / (results_ai + results_random)

        print("win-rate: {0}% and {1} draws".format(win_rate, results_draw))

        return win_rate