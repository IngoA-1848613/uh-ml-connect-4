from copy import deepcopy

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model

from lib.connectfour import Game
from models.model2 import DRAW
from preprocessing.preprocessor import Preprocessor

LOSS = "categorical_crossentropy"

TRAIN_SIZE = 0.7

EXTRA_FEATURES = 1
NUM_ROWS = 6
NUM_COLS = 7

INPUT_SIZE = (NUM_COLS * NUM_ROWS) + EXTRA_FEATURES
OUTPUT_SIZE = 3

LEARNING_RATE = 1e-3
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
LAYERS = [300, 225, 225, 225]

ACTIVATION_OUTPUT = "softmax"
ACTIVATION_HIDDEN = "relu"

BATCH_SIZE = 32
EPOCHS = 1


class Model3:
    _model: Sequential
    _dataset = []

    # Preprocessing
    def preprocess(self, dataset_name):
        preprocessor = Preprocessor(dataset_name)
        self._dataset = preprocessor.process_h()

    # Setters
    def set_dataset(self, dataset):
        self._dataset = dataset

    def set_parameters(self):
        pass

    # Input/output
    def get_input_output(self):
        input, output = [], []

        for data in self._dataset:
            board, results, features = data[0], data[1], data[2:]

            input.append(np.append(board, features))
            output.append(results)

        x = np.array(input).reshape((-1, INPUT_SIZE))
        y = to_categorical(output, OUTPUT_SIZE)

        return x, y

    def get_split_input_output(self):
        x, y = self.get_input_output()
        return train_test_split(x, y, train_size=TRAIN_SIZE)

    # Model
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

    # Training
    def train_model(self, name="model3", parameters=None):
        x_train, x_test, y_train, y_test = self.get_split_input_output()

        history = self._model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )

        self.save_model(name)
        return history

    def train_model_xy(self, x, y):
        return self._model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save/Load
    def save_model(self, name: str = "model3"):
        self._model.save("data/models/" + name)

    def load_model(self, name: str = "model3"):
        self._model = load_model("data/models/" + name)

    # Prediction
    def predict(self, board, features, pid):
        x = np.array(np.append(board, features)).reshape((-1, INPUT_SIZE))
        return self._model.predict(x)[0][pid]
        
    def predict_move(self, game: Game, starter: int, player: int, prev_move: int, legal_only: bool = True):
        best_column, highest_prediction = 3, 0

        for idx in range(NUM_COLS):
            if legal_only and not game.is_legal_move(idx):
                continue

            temp_game = deepcopy(game)
            temp_game.play_move(player, idx)

            player_id = 1 if player == 1 else 2
            pred_val = self.predict(board=temp_game.board, features=[starter], pid=player_id)

            if highest_prediction <= pred_val:
                best_column = idx
                highest_prediction = pred_val

        return best_column
