import numpy as np
import pandas as pd
from lib.connectfour import Game


class Converter:

    # Provided
    def as_board(self, moves):
        states = np.empty((len(moves), 7*6))
        labels = np.empty((len(moves), 2))

        game = Game()
        for _, i in moves.iterrows():
            idx, player, move, winner = i['idx'], i['player'], i['move'], i['winner']
            game.play_move(player, move)
            states[idx, :] = game.board.reshape((-1))
            labels[idx, 0] = winner
            labels[idx, 1] = player

        return (states, labels)

    def read_games(self, fname):
        X = pd.read_csv(
            fname, names=["game", "idx", "player", "move", "winner"])
        for _, game in X.groupby('game'):
            yield self.as_board(game)

    def create_default(self, input: str, output: str):
        X = np.vstack([np.hstack(game) for game in self.read_games(input)])
        np.save(output, X)

    @classmethod
    def get_default(self, item):
        board_size = 6*7
        return item[0:board_size], item[board_size:]

    # Custom (starter)
    def as_board_starter(self, moves):
        states = np.empty((len(moves), 7*6))
        labels = np.empty((len(moves), 3))

        game = Game()
        starter = None

        for _, i in moves.iterrows():
            idx, player, move, winner = i['idx'], i['player'], i['move'], i['winner']

            if starter is None:
                starter = player

            game.play_move(player, move)
            states[idx, :] = game.board.reshape((-1))

            labels[idx, 0] = winner
            labels[idx, 1] = player
            labels[idx, 2] = starter

        # format (states, labels)
        # states (for every move: representation of the board)
        # labels (for every move: winner, player, starter)

        return (states, labels)

    def read_games_starter(self, filename: str):
        X = pd.read_csv(
            filename, names=["game", "idx", "player", "move", "winner"])

        for _, game in X.groupby('game'):

            # format (game)
            #      game  idx  player  move  winner
            # 114     0    0       1     1      -1
            # 115     0    1      -1     3      -1
            # 116     0    2       1     6      -1
            # 117     0    3      -1     0      -1
            # 118     0    4       1     6      -1
            # 119     0    5      -1     4      -1
            # 120     0    6       1     5      -1
            # 121     0    7      -1     3      -1
            # 122     0    8       1     1      -1
            # 123     0    9      -1     4      -1
            # 124     0   10       1     6      -1
            # 125     0   11      -1     6      -1
            # 126     0   12       1     1      -1
            # 127     0   13      -1     1      -1
            # 128     0   14       1     0      -1
            # 129     0   15      -1     4      -1
            # 130     0   16       1     1      -1
            # 131     0   17      -1     4      -1

            yield self.as_board_starter(game)

    def create_starter(self, input: str, output: str):
        X = np.vstack([np.hstack(game)
                      for game in self.read_games_starter(input)])
        np.save(output, X)

    @classmethod
    def get_starter(self, item):
        board_size = 6*7
        return item[0:board_size], item[board_size:]
