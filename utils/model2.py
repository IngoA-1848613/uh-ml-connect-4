import os
from copy import deepcopy

import numpy as np
from lib.connectfour import Game
from sklearn.model_selection import KFold, train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, InputLayer
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

# Tensorflow: Only errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_COLS = 7
NUM_ROWS = 6

INPUT_SIZE = NUM_COLS*NUM_ROWS
OUTPUT_SIZE = NUM_COLS
OUTPUT_ACTIVATION = "softmax"
METRICS = ["accuracy"]

# LOSS = "mse"
LOSS = "categorical_crossentropy" # "kl_divergence"
OPTIMIZER = "adam"  # adam, adamax, nadam, rmsprop

HIDDEN_LAYERS = [400, 400, 200]
HIDDEN_ACTIVATION = "relu"

EPOCHS = 10
BATCH_SIZE = 16
TEST_SIZE = 0.2


PLAYER_RANDOM = -1
PLAYER_AI = 1
DRAW = 0


class Model2:
    _model: Sequential
    _dataset = []

    def __init__(self, dataset = []) -> None:
        self._dataset = dataset

    # Formatting
    def get_input_output(self):
        input, output = [], []

        for data in self._dataset:
            board, move = data

            input.append(board)
            output.append(move)

        X = np.array(input).reshape((-1, INPUT_SIZE))
        y = to_categorical(output, OUTPUT_SIZE)

        return X, y

    def get_split_input_output(self):
        X, y = self.get_input_output()
        return train_test_split(X, y, test_size=TEST_SIZE)

    # Model methods
    def create_model(self):
        self._model = Sequential()

        # add input layer (prevent over-fitting)
        # self._model.add(InputLayer(input_shape=(INPUT_SIZE,)))
        self._model.add(Dropout(0.05, input_shape=(INPUT_SIZE,)))

        # add hidden layers
        for num_neurons in HIDDEN_LAYERS:
            self._model.add(Dense(
                num_neurons, activation=HIDDEN_ACTIVATION))

        # add output layer
        self._model.add(Dense(OUTPUT_SIZE, activation=OUTPUT_ACTIVATION))

        # compile model
        self._model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

        return self._model

    def train_model(self):
        X_train, X_test, y_train, y_test = self.get_split_input_output()

        history = self._model.fit(X_train, y_train, validation_data=(
            X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

        self._model.save("data/models/model2")

        return history

    def train_model_xy(self, X, y):
        history = self._model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
        return history

    def load_model(self):
        self._model = load_model("data/models/model2")

    # Prediction
    def predict(self, board):
        input = np.array(board).reshape((-1, INPUT_SIZE))
        return self._model.predict(input)

    def predict_one(self, board):
        return self.predict([board])[0]

    def predict_move(self, game: Game, player: int):
        game_copy = deepcopy(game)
        board = game_copy.board

        moves = self.predict_one(board)

        for i in range(NUM_COLS):
            if not game.is_legal_move(i):
                moves[i] = 0

        best_move = np.argmax(moves)
        return best_move

    # Validation
    def accuracy(self, y_truth, y_prediction):
        score = 0

        for i in range(len(y_truth)):
            argmax_truth, argmax_prediction = np.argmax(
                y_truth[i]), np.argmax(y_prediction[i])

            if argmax_truth == argmax_prediction:
                score += 1

        return (score / len(y_truth)) * 100

    def cross_validation(self):
        X, y = self.get_input_output()

        splits = 5
        kf = KFold(n_splits=splits, shuffle=True)

        count = 0
        for train_range, test_range in kf.split(X):
            X_train, X_test = X[train_range], X[test_range]
            y_train, y_test = y[train_range], y[test_range]

            self.create_model()
            self.train_model_xy(X_train, y_train)

            accuracy = self.accuracy(y_test, self.predict(X_test))
            print("Accuracy for fold no. {0} is: {1}".format(count, accuracy))

            count += 1

    # Validation against (random, monte carlo)
    def validate_against_random(self):
        iterations = 100
        result_values = {PLAYER_AI: "ai",
                         PLAYER_RANDOM: "random", DRAW: "draw"}
        starts_values = {PLAYER_AI: "ai",
                         PLAYER_RANDOM: "random"}

        results = {PLAYER_AI: 0, PLAYER_RANDOM: 0, DRAW: 0}
        starts = {PLAYER_AI: 0, PLAYER_RANDOM: 0}

        for i in range(iterations):

            game = Game()

            active_player = PLAYER_AI if i < (
                iterations / 2) else PLAYER_RANDOM
            start_player = active_player

            while game.check_status() == None:
                move = 3

                if active_player == PLAYER_AI:
                    best_move = self.predict_move(game, active_player)
                    move = best_move
                else:
                    random_move = game.random_action(legal_only=True)
                    move = random_move

                game.play_move(player=active_player, column=move)
                active_player *= -1

            starts[start_player] += 1
            results[game.status] += 1

            print("iteration ({0}): win({1}) and starts({2})".format(
                i, result_values[game.status], starts_values[start_player]))

        results_ai = results[PLAYER_AI]
        results_random = results[PLAYER_RANDOM]
        results_draw = results[DRAW]

        win_rate = results_ai / (results_ai + results_random)

        print("win-rate: {0}% and {1} draws".format(win_rate, results_draw))

        return results, starts

    def validate_against_monte_carlo(self, n=5):
        iterations = 100
        result_values = {PLAYER_AI: "ai",
                         PLAYER_RANDOM: "random", DRAW: "draw"}
        starts_values = {PLAYER_AI: "ai",
                         PLAYER_RANDOM: "random"}

        results = {PLAYER_AI: 0, PLAYER_RANDOM: 0, DRAW: 0}
        starts = {PLAYER_AI: 0, PLAYER_RANDOM: 0}

        for i in range(iterations):

            game = Game()

            active_player = PLAYER_AI if i < (
                iterations / 2) else PLAYER_RANDOM
            start_player = active_player

            while game.check_status() == None:
                move = 3

                if active_player == PLAYER_AI:
                    best_move = self.predict_move(game, active_player)
                    move = best_move
                else:
                    random_move, _ = game.smart_action(player=active_player, n=n, legal_only=True)
                    move = random_move

                game.play_move(player=active_player, column=move)
                active_player *= -1

            starts[start_player] += 1
            results[game.status] += 1

            results_ai = results[PLAYER_AI]
            results_random = results[PLAYER_RANDOM]
            results_draw = results[DRAW]

            win_rate = results_ai / (results_ai + results_random)

            print("iteration ({0}): win({1}) and starts({2}) and win-rate ({3})".format(
                i, result_values[game.status], starts_values[start_player], win_rate))

        results_ai = results[PLAYER_AI]
        results_random = results[PLAYER_RANDOM]
        results_draw = results[DRAW]

        win_rate = results_ai / (results_ai + results_random)

        print("win-rate: {0}% and {1} draws".format(win_rate, results_draw))

        return results, starts

