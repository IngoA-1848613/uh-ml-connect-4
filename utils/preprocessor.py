import numpy as np
from utils.converter import Converter

SEED = 1850394


class Preprocessor:
    _filename: str

    def __init__(self, filename: str) -> None:
        self._filename = filename

    # input (board), output (winner)
    def process_bw(self):
        data, history = np.load(self._filename), []

        # shuffle data
        np.random.seed(SEED)
        np.random.shuffle(data)

        # add to history
        for item in data:
            board, output = Converter.get(item)
            _, winner, _, _ = output
            history.append((board, winner))

        return history

    # input (board), output (move)
    def process_bm(self):
        data, history = np.load(self._filename), []

        # shuffle data
        np.random.seed(SEED)
        np.random.shuffle(data)

        # add to history
        for item in data:
            board, output = Converter.get(item)
            player, winner, move, _ = output

            wins = player == winner

            if wins:
                if player == -1:
                    board = np.negative(board)
                    board[board==0.] = 0.

                history.append((board, move))

                # print(winner, move)
                # print(np.array(board).reshape((6, 7)))

        return history

