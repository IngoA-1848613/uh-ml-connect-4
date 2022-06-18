import numpy as np
import pandas as pd
from connectfour import Game

def as_board(moves):
    states = np.empty((len(moves), 7*6))
    labels = np.empty((len(moves), 2))

    game = Game()
    for _,i in moves.iterrows():
        idx, player, move, winner = i['idx'], i['player'], i['move'], i['winner']
        game.play_move(player, move)
        states[idx,:] = game.board.reshape((-1))
        labels[idx,0] = winner
        labels[idx,1] = player

    return (states, labels)

def read_games(fname):
    X = pd.read_csv(fname, names=["game", "idx", "player", "move", "winner"])
    for _, game in X.groupby('game'):
        yield as_board(game)

X = np.vstack([np.hstack(game) for game in read_games()])
np.save('c4.npy', X)
