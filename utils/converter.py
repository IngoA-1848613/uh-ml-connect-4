import numpy as np
import pandas as pd
from lib.connectfour import Game


class Converter:

    # General methods
    def read_games(self, input, callback):
        columns = ["game", "idx", "player", "move", "winner"]

        X = pd.read_csv(input, names=columns)
        for _, game in X.groupby('game'):
            yield callback(game)

    def create(self, input: str, output: str, callback):
        X = np.vstack([np.hstack(game) for game in self.read_games(input, callback)])
        np.save(output, X)
        print("converted: {0} to {1}".format(input, output))

    @classmethod
    def get(self, item):
        board_size = 6*7
        return item[0:board_size], item[board_size:]

    # Board parsers
    # Default
    @classmethod
    def board(self, moves):
        states = np.empty((len(moves), 7*6))
        labels = np.empty((len(moves), 4))

        game = Game()
        starter = None

        for _, i in moves.iterrows():
            idx, player, move, winner = i['idx'], i['player'], i['move'], i['winner']

            if starter is None:
                starter = player

            game.play_move(player, move)
            states[idx, :] = game.board.reshape((-1))

            labels[idx, 0] = player
            labels[idx, 1] = winner
            labels[idx, 2] = move
            labels[idx, 3] = starter

        return (states, labels)

    # Capture board before move
    @classmethod
    def board_bbm(self, moves):
        states = np.empty((len(moves), 7*6))
        labels = np.empty((len(moves), 4))

        game = Game()
        starter = None

        # moves.iter-rows() -> a list of moves between 2 players
        for _, i in moves.iterrows():
            idx, player, move, winner = i['idx'], i['player'], i['move'], i['winner']

            if starter is None:
                starter = player

            states[idx, :] = game.board.reshape((-1))
            game.play_move(player, move)

            labels[idx, 0] = player
            labels[idx, 1] = winner
            labels[idx, 2] = move
            labels[idx, 3] = starter

        return (states, labels)
