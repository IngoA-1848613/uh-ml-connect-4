
# Constants
from utils.converter import Converter
from utils.preprocessor import Preprocessor

SEED = 1850394

PATH = "data/csv/"
DS_10K = "c4-10k.csv"
DS_50K = "c4-50k.csv"
DATASET = PATH + DS_10K


# Main function
class Main:
    def convert(self):
        converter = Converter()
        converter.create_starter(DATASET, "data/dat/c4s-10k.npy")

    def preprocess(self):
        preprocessor = Preprocessor("data/dat/c4s-10k.npy")
        preprocessor.process_history_starter()

    def train(self):
        print("train")

    def test(self):
        print("test")


# Execute main
if __name__ == "__main__":
    main = Main()
    # main.convert()
    main.preprocess()
