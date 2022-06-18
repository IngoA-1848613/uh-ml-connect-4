import numpy as np

from utils.converter import Converter


class Preprocessor:
    _filename: str

    def __init__(self, filename: str) -> None:
        self._filename = filename

    def process_history_starter(self):
        data, history = np.load(self._filename), []

        for item in data[:2]:
            board, options = Converter.get_starter(item)
            winner, player, starter = options

            print(board, winner, player, starter)
