import numpy as np

from preprocessing.converter import Converter

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
            _, winner, _, _, _ = output
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
            player, winner, move, started, prev_move = output
            wins = player == winner

            if wins:
                if player == -1:
                    board = np.negative(board)
                    board[board == 0.] = 0

                last_move = np.array([0, 0, 0, 0, 0, 0, 0])
                if prev_move >= 0:
                    last_move[int(prev_move)] = 1

                history.append((board, np.append(np.append(board, started), last_move), move))

        return history
