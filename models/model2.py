import os
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from lib.connectfour import Game
# Tensorflow: Only errors
from preprocessing.preprocessor import Preprocessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_COLS = 7
NUM_ROWS = 6

INPUT_SIZE = NUM_COLS * (NUM_ROWS + 0) + 0
OUTPUT_SIZE = NUM_COLS
OUTPUT_ACTIVATION = "softmax"
METRICS = ["accuracy"]

# LOSS = "mse"
LOSS = "categorical_crossentropy"  # "kl_divergence"
OPTIMIZER = "adam"  # adam, adamax, nadam, rmsprop

HIDDEN_LAYERS = [200, 200]
HIDDEN_ACTIVATION = "relu"

EPOCHS = 5
BATCH_SIZE = 32
TEST_SIZE = 0.2
DROPOUT_RATE = 0.1

PLAYER_RANDOM = -1
PLAYER_AI = 1
DRAW = 0


class Model2:
    _model: Sequential
    _dataset = []

    # Preprocessing
    def preprocess(self, dataset_name):
        preprocessor = Preprocessor(dataset_name)
        self._dataset = preprocessor.process_bm()

    # Setters
    def set_dataset(self, dataset):
        self._dataset = dataset

    def set_parameters(self):
        pass

    # Formatting
    def get_input_output(self):
        input, output = [], []

        for data in self._dataset:
            input_data, output_data = data
            board, starter, player, move = input_data

            input.append(Preprocessor.convert_bm_board(board, starter, player, move))
            output.append(output_data)

        x = np.array(input)
        y = to_categorical(output, OUTPUT_SIZE)

        print(input[:5])

        return x, y

    def get_split_input_output(self):
        x, y = self.get_input_output()
        return train_test_split(x, y, test_size=TEST_SIZE)

    # Model methods
    def create_model(self):
        self._model = Sequential()

        # add input (flatten) layer
        self._model.add(Flatten())

        # add dropout layer (prevent over-fitting)
        self._model.add(Dropout(DROPOUT_RATE, input_shape=(INPUT_SIZE,)))

        # add hidden layers
        for num_neurons in HIDDEN_LAYERS:
            self._model.add(Dense(
                num_neurons, activation=HIDDEN_ACTIVATION))

        # add output layer
        self._model.add(Dense(OUTPUT_SIZE, activation=OUTPUT_ACTIVATION))

        # compile model
        self._model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

        return self._model

    def train_model(self, name: str = "model2"):
        x_train, x_test, y_train, y_test = self.get_split_input_output()

        history = self._model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        self.save_model(name)
        return history

    def train_model_xy(self, x, y):
        history = self._model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
        return history

    def save_model(self, name: str = "model2"):
        self._model.save("data/models/" + name)

    def load_model(self, name: str = "model2"):
        self._model = load_model("data/models/" + name)

    # Prediction
    def predict(self, x):
        x = np.array(x).reshape((-1, INPUT_SIZE))
        return self._model.predict(x)

    def predict_one(self, board):
        return self.predict([board])[0]

    def predict_move(self, game: Game, starter: int, player: int, prev_move: int):
        game_copy = deepcopy(game)
        board = np.array(game_copy.board).flatten()

        input_value = Preprocessor.convert_bm_board(board, starter, player, prev_move)
        moves = self.predict_one(input_value)

        for i in range(NUM_COLS):
            if not game.is_legal_move(i):
                moves[i] = 0

        best_move = np.argmax(moves)
        return best_move
