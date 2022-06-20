import tensorflow as tf

from preprocessing.converter import Converter
from models.model2 import Model2
from preprocessing.preprocessor import Preprocessor

# Constants
SEED = 1850394


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

        preprocessor = Preprocessor("data/dat/c4bbm-50k.npy")
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

        model.validate_against_monte_carlo(5)

        # n=5  -> 84%
        # n=10 -> 73%
        # n=40 -> 35%


# Execute main
if __name__ == "__main__":
    main = Main()

    # Actions
    # main.preprocess()
    # main.train()

    main.test_against_x()
