import tensorflow as tf

from models.model3 import Model3
from preprocessing.converter import Converter
from utils.validator import Validator

# Constants
SEED = 1850394


# Main function
class Main:
    dataset = []
    dataset_name = "data/dat/c4-50k.npy"
    model_name = "model3"
    model = Model3()

    # Init
    def __init__(self) -> None:
        print("initializing ...")
        tf.keras.utils.set_random_seed(SEED)

    # Preprocessing
    @staticmethod
    def convert():
        print("converting ...")

        converter = Converter()
        converter.create("data/csv/c4-10k.csv", "data/dat/c4-10k.npy")
        converter.create("data/csv/c4-50k.csv", "data/dat/c4-50k.npy")

    def preprocess(self):
        print("preprocessing ...")

        self.model.preprocess(self.dataset_name)

    # Training/evaluation
    def train(self):
        print("training ...")

        self.model.create_model()
        self.model.train_model()

        # , 32, 64, 128, 256
        # for batch_size in [256]:
        #     self.model.train_model(batch_size, "model_b_" + str(batch_size))

    @staticmethod
    def test():
        print("running cross_validation ...")
        # self.model.cross_validation()

    def test_against_game(self):
        print("testing against game ...")

        self.model.load_model()

        validator = Validator()
        validator.validate_against_game(self.model.predict_move, n=5, iterations=100)


# Execute main
if __name__ == "__main__":
    main = Main()

    # Actions
    # Main.convert()
    # main.preprocess()
    # main.train()
    #
    main.test_against_game()
