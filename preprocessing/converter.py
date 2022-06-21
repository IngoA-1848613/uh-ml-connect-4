import numpy as np
import pandas as pd

from lib.connectfour import Game


class Converter:

    # General methods
    @staticmethod
    def read_games(input_file: str):
        columns = ["game", "idx", "player", "move", "winner"]

        data = pd.read_csv(input_file, names=columns)
        for _, game in data.groupby('game'):
            yield Converter.board(game)

    @staticmethod
    def create(input_file: str, output_file: str):
        data = np.vstack([np.hstack(game) for game in Converter.read_games(input_file)])
        np.save(output_file, data)

        print("converted: {0} to {1}".format(input_file, output_file))

    @classmethod
    def get(cls, item):
        board_size = 6 * 7
        board_size2 = board_size * 2

        return item[0:board_size], item[board_size:board_size2], item[board_size2], item[board_size2 + 1], \
               item[board_size2 + 2], item[board_size2 + 3], item[board_size2 + 4]

    # Board parsers
    # Default
    @classmethod
    def board(cls, moves):
        board_before = np.empty((len(moves), 7 * 6))
        board_after = np.empty((len(moves), 7 * 6))
        label_starter = np.empty((len(moves), 1))
        label_player = np.empty((len(moves), 1))
        label_winner = np.empty((len(moves), 1))
        label_curr_move = np.empty((len(moves), 1))
        label_prev_move = np.empty((len(moves), 1))

        game = Game()

        starter = None
        prev_idx = 0
        prev_move = 3

        for i, row in moves.iterrows():
            idx, player, move, winner = row['idx'], row['player'], row['move'], row['winner']

            if starter is None:
                starter = player

            board_before[idx, :] = game.board.flatten()
            game.play_move(player, move)
            board_after[idx, :] = game.board.flatten()

            label_starter[idx, 0] = starter
            label_player[idx, 0] = player
            label_winner[idx, 0] = winner
            label_curr_move[idx, 0] = move
            label_prev_move[idx, 0] = -1

            if prev_idx != idx - 1:
                label_prev_move[idx, 0] = prev_move

            prev_idx = idx
            prev_move = move

        return board_before, board_after, label_starter, label_player, label_winner, label_curr_move, label_prev_move
