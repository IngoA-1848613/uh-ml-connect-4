import numpy as np

from preprocessing.converter import Converter

# Constants
SEED = 1850394


# Class
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
            board_before, board_after, starter, player, winner, curr_move, prev_move = Converter.get(item)
            history.append(((board_after, starter, player, curr_move), winner))

        return history

    # input (board), output (move)
    def process_bm(self):
        data, history = np.load(self._filename), []

        # shuffle data
        np.random.seed(SEED)
        np.random.shuffle(data)

        # add to history
        for item in data:
            board_before, board_after, starter, player, winner, curr_move, prev_move = Converter.get(item)
            wins = player == winner

            if wins:
                if player == -1:
                    board_before = np.negative(board_before)
                    board_before[board_before == 0.] = 0

                history.append(((board_before, starter, player, prev_move), curr_move))

        return history

    # input (board), output (history)
    def process_h(self):
        data, history = np.load(self._filename), []

        # shuffle data
        # np.random.seed(SEED)
        # np.random.shuffle(data)

        # add to history
        for item in data:
            board_before, board_after, starter, player, winner, curr_move, prev_move = Converter.get(item)
            history.append((board_after, winner, starter))

        return history

    # Converters
    @staticmethod
    def convert_bw(board, starter, player, prev_move):
        output = board
        return output

    @staticmethod
    def convert_bm_board(board, starter, player, prev_move):
        board = np.array(board).flatten()
        return board

    @staticmethod
    def convert_bm_board_move(board, starter, player, prev_move):
        board = np.array(board).flatten()
        last_move = [0, 0, 0, 0, 0, 0, 0]

        if prev_move != -1:
            last_move[prev_move] = 1

        return np.append(board, last_move)

    @staticmethod
    def convert_bm_board_starter(board, starter, player, prev_move):
        board = np.array(board).flatten()
        return np.append(board, starter)

    @staticmethod
    def convert_bm_board_move_starter(board, starter, player, prev_move):
        board = np.array(board).flatten()
        output_board_move = Preprocessor.convert_bm_board_move(board, starter, player, prev_move)
        return np.append(output_board_move, starter)
