import numpy as np
import time
import itertools


class Player:

    def __init__(self, name, inp, method='ideal'):

        self.inp = inp
        self.name = name
        self.method = method
        self.all_moves = np.arange(1, 10).reshape(3, 3)

    def get_valid_moves(self, board_state):
        '''returns a list of valid moves'''
        valid_moves = [ind+1 for ind, val in enumerate(board_state.reshape(-1))
                       if val == 0]
        return valid_moves

    def find_move_rand(self, board_state):
        '''choose a random valid move'''
        valid_moves = self.get_valid_moves(board_state)
        return np.random.choice(valid_moves)

    def find_winning_move(self, board_state):
        next_move = None
        for i in range(3):
            if np.sum(board_state[i]) == 2:
                next_move = np.intersect1d(self.all_moves[i],
                                           self.valid_moves)[-1]
            elif np.sum(board_state[:, i]) == 2:
                next_move = np.intersect1d(self.all_moves[:, i],
                                           self.valid_moves)[-1]
            elif np.sum(board_state.diagonal()) == 2:
                next_move = np.intersect1d(self.all_moves.diagonal(),
                                           self.valid_moves)[-1]
            elif np.sum(np.fliplr(board_state).diagonal()) == 2:
                next_move = np.intersect1d(
                    np.fliplr(self.all_moves).diagonal(),
                    self.valid_moves)[-1]
            if next_move:
                break
        return next_move

    def find_forking_move(self, board_state):
        next_move = None
        # find forks along lines and rows
        for i, j in itertools.combinations(range(3), 2):
            if (np.sum(board_state[i]) == 1 and np.sum(board_state[:, j]) == 1
                    and i != j):
                next_move = np.intersect1d(self.all_moves[i],
                                           self.valid_moves)[-1]
            if next_move:
                break

        # if no forks are found try to find forks along diagonals
        if not next_move:
            if ((np.sum(board_state[0, 1:]) == 1
                 or np.sum(board_state[2, :1]) == 1)
                    and np.sum(board_state.diagonal()) == 1):
                next_move = np.intersect1d(self.all_moves.diagonal(),
                                           self.valid_moves)[-1]
            elif ((np.sum(board_state[0, :1]) == 1
                   or np.sum(board_state[2, 1:]) == 1)
                  and np.sum(np.fliplr(board_state).diagonal()) == 1):
                next_move = np.intersect1d(
                    np.fliplr(self.all_moves).diagonal(),
                    self.valid_moves)[-1]
        return next_move

    def find_opposite_corner_move(self, board_state):
        next_move = None
        for i, j in itertools.combinations([0, -1], 2):
            if board_state[i, j] == -1:
                anti_i = (i+1)*-1
                anti_j = (j+1)*-1
                next_move = np.intersect1d(self.all_moves[anti_i, anti_j],
                                           self.valid_moves)[-1]
                break
        return next_move

    def find_move_ideal(self, board_state):
        '''find ideal move based on the following priorities:
        1. Win
        2. Block
        3. Fork
        4. Block Fork
        5. Center
        6. Opposite Corner to opponent
        7. Empty Corner
        8. Empty Side'''
        self.valid_moves = self.get_valid_moves(board_state)
        # make board the same for every player (1 vs -1) so that we can always
        # consider 1 to be the player considering the moves
        dummy_board_state = board_state*self.inp
        win_move = self.find_winning_move(dummy_board_state)
        if win_move:
            return win_move
        block_move = self.find_winning_move(-1*dummy_board_state)
        if block_move:
            return block_move
        fork_move = self.find_forking_move(dummy_board_state)
        if fork_move:
            return fork_move
        block_fork_move = self.find_forking_move(-1*dummy_board_state)
        if block_fork_move:
            return block_fork_move
        center_move = np.intersect1d([5], self.valid_moves)
        if center_move:
            return center_move
        opposite_corner_move = self.find_opposite_corner_move(
            dummy_board_state)
        if opposite_corner_move:
            return opposite_corner_move
        empty_corner_move = np.random.choice(np.intersect1d([1, 3, 7, 9],
                                                            self.valid_moves))
        if empty_corner_move:
            return empty_corner_move
        empty_side_move = self.find_move_rand(self, board_state)
        if empty_side_move:
            return empty_side_move

    def find_move_manual(self):
        '''let players choose their moves'''
        return int(input('Make a move {}: '.format(self.name)))-1

    def move(self, board_state):
        if self.method == 'ideal':
            self.next_move = self.find_move_ideal(board_state)-1
        elif self.method == 'rand':
            self.next_move = self.find_move_rand(board_state)
        elif self.method == 'manual':
            self.next_move = self.find_move_manual()
        return self.next_move, self.inp


class Board:

    length = 3
    width = 3

    def __init__(self, Players):

        self.board_state = np.zeros((self.length, self.width), dtype=int)
        self.board_state_visual = np.array(range(1, 10),
                                           dtype=str).reshape(3, 3)
        self.rounds = 0
        self.moves = []
        self.over = False
        self.Players = Players
        self.symbols = ['o', 'x']

    def move(self, pos, inp):

        self.board_state = self.board_state.reshape(-1)
        self.board_state_visual = self.board_state_visual.reshape(-1)
        self.board_state[pos] = inp
        self.board_state_visual[pos] = self.symbols[int(inp/2+.5)]
        self.board_state = self.board_state.reshape(3, 3)
        self.board_state_visual = self.board_state_visual.reshape(3, 3)
        self.moves.append(pos)

    def end_game(self):

        print('{} wins!'.format(self.Winner))
        time.sleep(1)

    def win_cond(self, board_state):
        cond = ((np.sum(board_state, axis=0) == 3).any()
                or (np.sum(board_state, axis=1) == 3).any()
                or np.sum(board_state.diagonal()) == 3
                or np.sum(np.fliplr(board_state).diagonal) == 3)
        return cond

    def check_win(self):
        for i in range(3):
            if self.win_cond(self.board_state):
                self.over = True
                self.Winner = 'Player 1'
            elif self.win_cond(self.board_state*-1):
                self.over = True
                self.Winner = 'Player 2'
        if len(self.moves) == 9:
            self.Winner = 'Nobody'
            self.over = True

    def visualize(self):
        for i in range(len(self.board_state)):
            if i != 0:
                print('---+---+---')
            print(' {} | {} | {} '.format(*self.board_state_visual[i]))
        print('\n\n')

    def round(self, pos, inp):

        self.move(pos, inp)
        self.visualize()
        self.check_win()

    def main(self):

        self.visualize()
        while not self.over:
            for Player in self.Players:
                while True:
                    try:
                        pos, inp = Player.move(self.board_state)
                        if pos in self.moves:
                            raise ValueError
                    except ValueError:
                        print('Valid move, please!')
                        time.sleep(10)
                        continue
                    break

                self.round(pos, inp)
                if self.over:
                    self.end_game()
                    break
