import os
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from lib.connectfour import Game

# Tensorflow: Only errors
from preprocessing.preprocessor import Preprocessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_COLS = 7
NUM_ROWS = 6

INPUT_SIZE = NUM_COLS * NUM_ROWS
OUTPUT_SIZE = 3
OUTPUT_ACTIVATION = "softmax"
METRICS = ["accuracy"]

LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"  # rmsprop

HIDDEN_LAYERS = [200, 200]
HIDDEN_ACTIVATION = "relu"

EPOCHS = 15
BATCH_SIZE = 32
TEST_SIZE = 0.2


class Model:
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
            board, winner = data

            input.append(board)
            output.append(winner)

        X = np.array(input).reshape((-1, INPUT_SIZE))
        y = to_categorical(output, 3)

        return X, y

    def get_split_input_output(self):
        x, y = self.get_input_output()
        return train_test_split(x, y, test_size=TEST_SIZE)

    # Model methods
    def create_model(self):
        self._model = Sequential()

        # add input layer (prevent over-fitting)
        self._model.add(Dropout(0.1, input_shape=(INPUT_SIZE,)))

        # add hidden layers
        for num_neurons in HIDDEN_LAYERS:
            self._model.add(Dense(num_neurons, activation=HIDDEN_ACTIVATION))

        # add output layer
        self._model.add(Dense(OUTPUT_SIZE, activation=OUTPUT_ACTIVATION))

        # compile model
        self._model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

        return self._model

    def train_model(self, name="model"):
        x_train, x_test, y_train, y_test = self.get_split_input_output()

        history = self._model.fit(x_train, y_train, validation_data=(
            x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

        self.save_model(name)

        return history

    def train_model_xy(self, x, y):
        return self._model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    def save_model(self, name: str = "model"):
        self._model.save("data/models/" + name)

    def load_model(self, name: str = "model"):
        self._model = load_model("data/models/" + name)

    # Prediction
    def predict(self, x):
        x = np.array(x).reshape((-1, INPUT_SIZE))
        return self._model.predict(x)

    def predict_one(self, input_value):
        return self.predict([input_value])[0]

    def predict_move(self, game: Game, starter: int, player: int, prev_move: int):

        best_move = 3,
        highest_prediction = 0

        for i in range(NUM_COLS):
            if not game.is_legal_move(i):
                continue

            game_copy = deepcopy(game)
            game_copy.play_move(player, i)

            prediction_value = self.predict_one(game_copy.board)[
                1 if player == 1 else 2]

            if prediction_value > highest_prediction:
                best_move = i
                highest_prediction = prediction_value

        return best_move
