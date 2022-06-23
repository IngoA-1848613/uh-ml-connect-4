import tensorflow as tf
import os

from models.model2 import Model2
from models.model3 import Model3
from preprocessing.converter import Converter
from utils.validator import Validator

# Constants
SEED = 1850394


# Main function
class Main:
    dataset = []
    dataset_name = "data/c4-50k.npy"
    model_name = "model_e_5"
    model = Model3()

    # Init
    def __init__(self) -> None:
        print("initializing ...")
        tf.keras.utils.set_random_seed(SEED)

    @staticmethod
    def set_seed():
        tf.keras.utils.set_random_seed(SEED)

    # Preprocessing
    @staticmethod
    def convert():
        print("converting ...")

        converter = Converter()
        converter.create("data/c4-50k.csv", "data/c4-50k.npy")

    def preprocess(self):
        print("preprocessing ...")

        Main.set_seed()
        self.model.preprocess(self.dataset_name)

    # Training/evaluation
    def train(self):
        print("training ...")

        Main.set_seed()
        self.model.create_model()

        Main.set_seed()
        self.model.train_model(name="model3")


    def test(self):
        print("running cross_validation ...")

        Validator.cross_validation_against_game(self.model)

    def test_against_game(self):
        print("testing against game ...")

        validator = Validator()

        Main.set_seed()
        self.model.load_model("model3")
        validator.validate_against_game(self.model.predict_move, n=5, iterations=100)

# Execute main
if __name__ == "__main__":
    main = Main()

    # Actions
    Main.convert()
    main.preprocess()
    main.train()
    main.test_against_game()
