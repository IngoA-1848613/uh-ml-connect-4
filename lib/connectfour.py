from random import random, randint
from copy import copy, deepcopy
import numpy as np
from sys import stderr

def check_consecutive(board, consecutive=4):
    for row in board:
        count = 1
        for col in range(1, len(row)):
            if row[col - 1] == row[col]:
                count += 1
            else:
                count = 1
            if count == consecutive and row[col] != 0:
                return row[col]
    return 0

def diagonals(L):
    h, w = len(L), len(L[0])
    return [[L[h - p + q - 1][q] for q in range(max(p-h+1, 0), min(p+1, w))] for p in range(h + w - 1)]

def antidiagonals(L):
    h, w = len(L), len(L[0])
    return [[L[p - q][q] for q in range(max(p-h+1,0), min(p+1, w))] for p in range(h + w - 1)]

def argmax(x, key=lambda x: x):
    (k, i, v) = max(((key(v), i, v) for i,v in enumerate(x)))
    return (i, v)

def starting_player():
    return 1 if randint(0, 1) == 0 else -1

class Game:
    width = 7
    height = 6
    consecutive = 4

    def __init__(self, board=None):
        if board is None:
            self.board = np.zeros((self.height, self.width), dtype='int')
        else:
            self.board = np.array(board)
        self.status = self.check_status()

    def random_action(self, legal_only=True):
        column = randint(0, self.width - 1)
        if legal_only:
            while not self.is_legal_move(column):
                column = randint(0, self.width - 1)
        return column

    def is_full(self):
        return np.all(self.board != 0)

    def is_legal_move(self, column):
        for row in range(self.height):
            if self.board[row, column] == 0:
                return True
        return False

    def play_move(self, player, column):
        legal_move = False
        for row in range(self.height):
            if self.board[row, column] == 0:
                self.board[row, column] = player
                legal_move = True
                break

        if not legal_move:
            status = player * -1
        else:
            status = self.check_status()
        self.status = status

    def check_status(self):
        h = check_consecutive(self.board, self.consecutive)
        if h != 0:
            return h

        v = check_consecutive(self.board.T, self.consecutive)
        if v != 0:
            return v

        d = check_consecutive(diagonals(self.board), self.consecutive)
        if d != 0:
            return d

        ad = check_consecutive(antidiagonals(self.board), self.consecutive)
        if ad != 0:
            return ad

        if self.is_full():
            return 0

        return None

    def __repr__(self):
        return repr(self.board)

    def random_play(self, starting=None, legal_only=True, f=None):
        player = starting if starting is not None else starting_player()

        f is not None and f(self, None, None)
        while self.status is None:
            move = self.random_action(legal_only=legal_only)
            self.play_move(player, move)
            f is not None and f(self, player, move)
            player = player * -1
        return self.status

    def winning(self, player, legal_only=True, n=1000):
        other = player * -1
        p = np.empty(self.width)
        for col in range(self.width):
            wins = losses = 0
            for i in range(n):
                game = deepcopy(self)
                game.play_move(player, col)
                status = game.random_play(other, legal_only=legal_only)
                if status == player:
                    wins += 1
                elif status != 0:
                    losses += 1

            ratio = (wins / losses) if losses != 0 else wins
            p[col] = ratio
        return p

    def smart_action(self, player, legal_only=True, n=100):
        p = self.winning(player, legal_only=legal_only, n=n)
        return argmax(p)

    def smart_play(self, starting=None, legal_only=True, n=100, f=None):
        player = starting if starting is not None else starting_player()

        f is not None and f(self, None, None)
        while self.status is None:
            move, p = self.smart_action(player, legal_only=legal_only, n=n)
            if not self.is_legal_move(move):
                print("illegal move", player, p, move, file=stderr)
            self.play_move(player, move)
            f is not None and f(self, player, move)
            player = player * -1
        return self.status



#game = Game([[ 0, -1,  1,  1, -1,  0,  0],
#       [ 0,  0,  0,  1, -1,  0,  0],
#       [ 0,  0,  0,  1,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0]])
#print(game.status, game)
#print(game.winning(-1, n=1000))

if __name__ == "__main__":
    from io import StringIO
    from mpi4py.futures import MPIPoolExecutor, MPICommExecutor

    def play_game(gameid, starting=None, legal_only=False):
        states = []
        def add_state(game, player, move):
            if player is not None and move is not None:
                states.append((player, move))

        game = Game()
        #status = game.random_play(starting, legal_only=legal_only, f=add_state)
        status = game.smart_play(starting, legal_only=legal_only, n=10, f=add_state)

        io = StringIO()
        for idx, move  in enumerate(states):
            print(gameid, idx, move[0], move[1], status, sep=",", file=io)
        return io.getvalue()

    def repeat(x):
      while True:
        yield x

    with MPICommExecutor() as pool:
        for result in pool.map(play_game, range(50000), unordered=True):
            print(result)
