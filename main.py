import tensorflow as tf

from models.model2 import Model2
from preprocessing.converter import Converter
from preprocessing.preprocessor import Preprocessor

# Constants
SEED = 1850394


# Main function
class Main:
    dataset = []
    dataset_name = "data/dat/c4bbm-50k.npy"
    model = Model2()

    # Init
    def __init__(self) -> None:
        print("initializing ...")
        tf.keras.utils.set_random_seed(SEED)

    # Preprocessing
    @staticmethod
    def convert():
        print("converting ...")

        converter = Converter()
        converter.create("data/csv/c4-10k.csv", "data/dat/c4-10k.npy", Converter.board)
        converter.create("data/csv/c4-50k.csv", "data/dat/c4-50k.npy", Converter.board)
        converter.create("data/csv/c4-10k.csv", "data/dat/c4bbm-10k.npy", Converter.board_bbm)
        converter.create("data/csv/c4-50k.csv", "data/dat/c4bbm-50k.npy", Converter.board_bbm)

    def preprocess(self):
        print("preprocessing ...")

        preprocessor = Preprocessor(self.dataset_name)
        self.dataset = preprocessor.process_bm()
        self.model.set_dataset(self.dataset)

    # Training/evaluation
    def train(self):
        print("training ...")

        self.model.set_dataset(self.dataset)
        self.model.create_model()
        self.model.train_model()

    def test(self):
        print("running cross_validation ...")

        self.model.set_dataset(self.dataset)
        self.model.cross_validation()

    def test_against_x(self):
        print("testing against player ...")

        self.model.load_model()
        self.model.validate_against_monte_carlo(5)
        self.model.validate_against_monte_carlo(10)
        self.model.validate_against_monte_carlo(40)

        # n=5  -> 84%
        # n=10 -> 73%
        # n=40 -> 35%


# Execute main
if __name__ == "__main__":
    main = Main()

    # Actions
    # Main.convert()
    main.preprocess()
    main.train()

    main.test_against_x()
