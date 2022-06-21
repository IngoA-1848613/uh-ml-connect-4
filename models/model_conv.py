import os
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from lib.connectfour import Game

# Tensorflow: Only errors
from preprocessing.preprocessor import Preprocessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_COLS = 7
NUM_ROWS = 6

INPUT_SIZE = NUM_COLS * NUM_ROWS
OUTPUT_SIZE = NUM_COLS
OUTPUT_ACTIVATION = "softmax"
METRICS = ["accuracy"]

# LOSS = "mse"
LOSS = "categorical_crossentropy"  # "kl_divergence"
OPTIMIZER = "adam"  # adam, adamax, nadam, rmsprop

HIDDEN_LAYERS = [200, 200]
HIDDEN_ACTIVATION = "relu"

EPOCHS = 15
BATCH_SIZE = 32
TEST_SIZE = 0.2
DROPOUT_RATE = 0.1

PLAYER_RANDOM = -1
PLAYER_AI = 1
DRAW = 0


class ModelConv:
    _model: Sequential
    _dataset = []

    # Preprocessing
    def preprocess(self, dataset_name):
        preprocessor = Preprocessor(dataset_name)
        self._dataset = preprocessor.process_bw()

    # Setters
    def set_dataset(self, dataset):
        self._dataset = dataset

    def set_parameters(self):
        pass

    # Formatting
    def get_input_output(self):
        input, output = [], []

        for data in self._dataset:
            board, move = data
            b = np.array(board).reshape((6, 7, 1))

            input.append(b)
            output.append(move)

        X = np.array(input)
        y = to_categorical(output, OUTPUT_SIZE)

        return X, y

    def get_split_input_output(self):
        X, y = self.get_input_output()
        return train_test_split(X, y, test_size=TEST_SIZE)

    # Model methods
    def create_model(self):
        self._model = Sequential()

        # Convolution
        self._model.add(Conv2D(64, (3, 3), input_shape=(6, 7, 1)))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the input
        self._model.add(Flatten())

        # Dropout to prevent over-fitting
        self._model.add(Dropout(DROPOUT_RATE, input_shape=(INPUT_SIZE,)))

        # add hidden layers
        for num_neurons in HIDDEN_LAYERS:
            self._model.add(Dense(num_neurons, activation=HIDDEN_ACTIVATION))

        # add output layer
        self._model.add(Dense(OUTPUT_SIZE, activation=OUTPUT_ACTIVATION))

        # compile model
        self._model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

        return self._model

    def train_model(self, name: str = "model_conv"):
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

    def train_model_xy(self, X, y):
        history = self._model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
        return history

    # Save/Load
    def save_model(self, name: str = "model_conv"):
        self._model.save("data/models/" + name)

    def load_model(self, name: str = "model_conv"):
        self._model = load_model("data/models/" + name)

    # Prediction
    def predict(self, x):
        x = np.array(x).reshape((-1, 6, 7, 1))
        return self._model.predict(x)

    def predict_one(self, input_value):
        return self.predict([input_value])[0]

    def predict_move(self, game: Game, starter: int, player: int, prev_move: int):
        game_copy = deepcopy(game)
        board = game_copy.board

        moves = self.predict_one(board)

        for i in range(NUM_COLS):
            if not game.is_legal_move(i):
                moves[i] = 0

        best_move = np.argmax(moves)
        return best_move
