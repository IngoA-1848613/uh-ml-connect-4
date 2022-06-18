import numpy as np
import tensorflow as tf
from utils.converter import Converter
from utils.model import Model
from utils.model2 import Model2
from utils.preprocessor import Preprocessor


# Constants
SEED = 1850394

PATH = "data/csv/"
DS_10K = "c4-10k.csv"
DS_50K = "c4-50k.csv"
DATASET = PATH + DS_50K


# Main function
class Main:
    dataset = []

    # Init
    def __init__(self) -> None:
        print("initializing ...")
        tf.keras.utils.set_random_seed(SEED)

    # Preprocessing
    def convert(self):
        print("converting ...")

        converter = Converter()
        converter.create("data/csv/c4-10k.csv", "data/dat/c4-10k.npy", Converter.board)
        converter.create("data/csv/c4-50k.csv", "data/dat/c4-50k.npy", Converter.board)
        converter.create("data/csv/c4-10k.csv", "data/dat/c4bbm-10k.npy", Converter.board_bbm)
        converter.create("data/csv/c4-50k.csv", "data/dat/c4bbm-50k.npy", Converter.board_bbm)

    def preprocess(self):
        print("preprocessing ...")
        # preprocessor = Preprocessor("data/dat/c4-10k.npy")
        # self.dataset = preprocessor.process_bw()

        preprocessor = Preprocessor("data/dat/c4bbm-10k.npy")
        self.dataset = preprocessor.process_bm()

    # Training/evaluation
    def train(self):
        print("training ...")

        model = Model2(self.dataset)
        model.create_model()
        model.train_model()

    def test(self):
        print("running cross_validation ...")

        model = Model2(self.dataset)
        model.cross_validation()

    def test_against_x(self):
        print("testing against player ...")

        model = Model2()
        model.load_model()

        # model.validate_against_random()
        model.validate_against_monte_carlo(5)


# Execute main
if __name__ == "__main__":
    main = Main()

    # Actions
    main.test_against_x()

