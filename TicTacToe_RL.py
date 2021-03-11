import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle, PathPatch


def Cross(center, radius, facecolor='none',
          edgecolor='b', linewidth=5, alpha=1):
    # create a cross patch to plot
    center = np.array(center[::-1])
    edge1 = center+[radius, radius]
    edge2 = center+[-radius, -radius]
    edge3 = center+[radius, -radius]
    edge4 = center+[-radius, radius]
    path = Path(np.array([edge1, edge2, center, edge3, edge4]))
    return PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                     linewidth=linewidth, alpha=alpha)


class Board:
    def __init__(self, p1, p2, p3=None, visual=False, save=False):
        self.visual = visual
        self.state = np.zeros((3, 3), dtype=np.int64)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.Players = [self.p1, self.p2]
        self.over = False
        self.symbol = 1
        self.results = []
        self.save_history = save

    def encode_state(self):
        # needs to be used as key for dictionary
        self.encoded_state = self.state.flatten().tobytes()
        return self.encoded_state

    def decode_state(self, encoded_state):
        # useful to get humanly readable state
        return np.frombuffer(encoded_state, dtype=np.int64)

    def win_cond(self, marker=1):
        cond = ((np.sum(self.state, axis=0) == 3*marker).any()
                or (np.sum(self.state, axis=1) == 3*marker).any()
                or np.sum(self.state.diagonal()) == 3*marker
                or np.sum(np.fliplr(self.state).diagonal()) == 3*marker)
        return cond

    def check_win(self):
        for i in range(3):
            if self.win_cond(marker=1):
                self.over = True
                self.Winner = self.Players[0]
                self.Loser = self.Players[1]
                return 1

            elif self.win_cond(marker=-1):
                self.over = True
                self.Winner = self.Players[1]
                self.Loser = self.Players[0]
                return -1

        # tie
        if len(self.valid_moves()) == 0:
            self.Winner = None
            self.over = True
            return 2
        self.over = False
        return None

    def valid_moves(self):
        moves = [ind for ind, val in np.ndenumerate(self.state) if val == 0]
        return moves

    def move(self, pos):
        self.state[pos] = self.symbol
        # switch player
        self.symbol *= -1

    def reward(self):
        result = self.check_win()
        if result == 2:
            self.p1.give_reward(.2)
            self.p2.give_reward(.3)

        else:
            self.Winner.give_reward(1)
            self.Loser.give_reward(0)

    def train(self, max_rounds=50):
        for round in range(max_rounds):
            print('Runde {}'.format(round))

            while not self.over:
                for Player in self.Players:
                    # print(self.symbol)
                    moves = self.valid_moves()
                    # get players move
                    move = Player.move(moves, self.state, self.symbol)
                    self.move(move)
                    # return encoded state to player
                    Player.add_state(self.encode_state())
                    # check for win
                    if self.check_win():
                        self.reward()
                        break
            self.p1.reset()
            self.p2.reset()
            # time.sleep(10)
            self.save()
            self.reset()
        for Player in self.Players:
            Player.save_training()

    def visualize(self):
        for i in range(len(self.state)):
            if i != 0:
                print('---+---+---')
            print(' {} | {} | {} '.format(*self.state[i]))
        print('\n\n')

    def save(self):
        if self.Winner:
            self.results.append(self.Winner.name)
        else:
            self.results.append('Tie')

    def reset(self):
        if self.Winner:
            if self.visual:
                self.visualize()
            print('{} has won!'.format(self.Winner))
        else:
            print('It is a draw!')
        self.state = np.zeros((3, 3), dtype=np.int64)
        self.over = False
        self.symbol = 1

    def play(self, max_rounds):
        for round in range(max_rounds):
            print('Runde {}'.format(round))

            while not self.over:
                for Player in self.Players:
                    if self.visual:
                        self.visualize()
                    # print(self.symbol)
                    moves = self.valid_moves()
                    # get players move
                    move = Player.move(moves, self.state, self.symbol)
                    self.move(move)
                    # return encoded state to player
                    Player.add_state(self.encode_state())
                    # check for win
                    if self.check_win():
                        break
            self.p1.reset()
            self.p2.reset()
            # time.sleep(10)
            self.save()
            self.reset()

    def alternate_train_play(self, batch_size, max_rounds):
        # alternate between training and playing against perfect opponent to
        # see the progress made
        epsilon = self.p1.get_exploration()
        for batch in range(0, max_rounds, batch_size):
            self.p1.change_exploration(epsilon)
            self.Players = [self.p1, self.p2]
            self.train(max_rounds=batch_size)
            if self.save_history:
                self.save_history('train_{:05}'.format(batch))
            # let trained agent play his best moves
            self.p1.change_exploration(.0)
            self.Players = [self.p1, self.p3]
            self.play(max_rounds=1)
            if self.save_history:
                self.save_history('play_{:05}'.format(batch))

    def save_history(self, name):
        print(name)
        with open('history_{}.csv'.format(name), 'w') as csv:
            np.savetxt(csv, self.results, fmt='%s', delimiter=',')
        self.results = []


class Agent:
    def __init__(self, name, epsilon=.3, lr=.2, gamma=.9):
        self.name = name
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma  # discount factor
        self.states_played = []
        self.state_values = {}  # dict to store state-value pairs

    def __repr__(self):
        return 'Agent {}'.format(self.name)

    def change_exploration(self, epsilon):
        # manually change exploration factor
        self.epsilon = epsilon

    def get_exploration(self):
        return self.epsilon

    def move(self, legal_moves, current_state, symbol):
        if np.random.rand() < self.epsilon:
            # random move
            move = legal_moves[np.random.randint(len(legal_moves))]
        else:
            best_value = -100
            # find action that leads to bestvalued next state
            for legal_move in legal_moves:
                next_state = current_state.copy()
                next_state[legal_move] = symbol
                key = next_state.tobytes()
                value = 0 if not self.state_values.get(key) \
                    else self.state_values.get(key)
                if value >= best_value:
                    best_value = value
                    move = legal_move
        return move

    def add_state(self, encoded_state):
        self.states_played.append(encoded_state)

    def give_reward(self, reward):
        # update values of all played states
        for state in self.states_played[:-1]:
            # update state value
            if not self.state_values.get(state):
                self.state_values[state] = 0
            self.state_values[state] += self.lr * \
                (self.gamma*reward - self.state_values[state])
            # use as new reward for previous state
            reward = self.state_values[state]

    def reset(self):
        # clear all round specific values
        self.states_played = []

    def save_training(self):
        with open('Training_{}'.format(self.name), 'wb') as file:
            pickle.dump(self.state_values, file)

    def load_training(self, path):
        with open('Training_{}'.format(path), 'rb') as file:
            self.state_values = pickle.load(file)

    def get_state_action_values(self, state, symbol):
        # get the values for all possible actions
        values = []
        moves = [ind if val == 0 else np.nan
                 for ind, val in np.ndenumerate(state)]
        for move in moves:
            if np.isnan(move).any():
                value = np.nan
            else:
                next_state = state.copy()
                next_state[move] = symbol
                key = next_state.tobytes()
                value = 0 if not self.state_values.get(key) \
                    else self.state_values.get(key)

            values.append(value)
        return np.array(values)

    def plot_state_action_values(self, state, symbol, path='state_action.png'):
        # make image showing the state-action pairs
        # also show curent state
        values = self.get_state_action_values(state, symbol)
        values = values.reshape((3, 3))
        fig, ax = plt.subplots(1, 1)
        crosses = [Cross(ind, .4, facecolor='none',
                         edgecolor='b', linewidth=5, alpha=1)
                   for ind, val in np.ndenumerate(state) if val == 1]
        circles = [Circle(ind, .4, facecolor='none',
                          edgecolor='r', linewidth=5, alpha=1)
                   for ind, val in np.ndenumerate(state) if val == -1]
        for cross in crosses:
            ax.add_patch(cross)
        for circle in circles:
            ax.add_patch(circle)
        im = ax.imshow(values)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for i in range(3):
            for j in range(3):
                if not np.isnan(values[i, j]):
                    ax.text(j, i, '{:.2f}'.format(values[i, j]),
                            ha="center", va="center", color="w")
        fig.colorbar(im, cmap='viridis')
        fig.savefig(path)


class Player:
    def __init__(self, name, method='manual'):
        self.name = name
        self.method = method
        self.moves = [(i, j) for i in range(3) for j in range(3)]
        self.all_moves = np.arange(1, 10).reshape(3, 3)

    def __repr__(self):
        return 'Player {}'.format(self.name)

    def move(self, legal_moves, current_state, symbol):
        if self.method == 'manual':
            move = self.move_manual(legal_moves, current_state, symbol)

        else:
            move = self.move_ideal(legal_moves, current_state, symbol)

        return move

    def move_ideal(self, legal_moves, current_state, symbol):
        '''find ideal move based on the following priorities:
        1. Win
        2. Block
        3. Fork
        4. Block Fork
        5. Center
        6. Opposite Corner to opponent
        7. Empty Corner
        8. Empty Side'''
        # make board the same for every player (1 vs -1) so that we can always
        # consider 1 to be the player considering the moves
        dummy_board_state = current_state*symbol
        # to adapt to different indexing of moves
        valid_moves = np.arange(1, 10).reshape(
            3, 3)[tuple(np.array(legal_moves).T)]
        win_move = self.find_winning_move(valid_moves, dummy_board_state)
        if win_move:
            return self.moves[win_move-1]
        block_move = self.find_winning_move(valid_moves, -1*dummy_board_state)
        if block_move:
            return self.moves[block_move-1]
        fork_move = self.find_forking_move(valid_moves, dummy_board_state)
        if fork_move:
            return self.moves[fork_move-1]
        # special case, need to block double fork
        if (dummy_board_state[1, 1] == 1
                and (np.sum(dummy_board_state.diagonal()) == -1
                     or np.sum(
                     np.fliplr(dummy_board_state).diagonal()) == -1)):
            return self.moves[np.random.choice([1, 3, 5, 8])]
        else:
            block_fork_move = self.find_forking_move(valid_moves,
                                                     -1*dummy_board_state)
        if block_fork_move:
            return self.moves[block_fork_move-1]
        center_move = np.intersect1d(
            [5], valid_moves)[-1] if np.intersect1d([5], valid_moves) else None
        if center_move:
            return self.moves[center_move-1]
        opposite_corner_move = self.find_opposite_corner_move(
            valid_moves,
            dummy_board_state)
        if opposite_corner_move:
            return self.moves[opposite_corner_move-1]
        empty_corner_move = np.intersect1d([1, 3, 7, 9], valid_moves)[-1] if \
            np.intersect1d([1, 3, 7, 9], valid_moves).any() else None
        if empty_corner_move:
            return self.moves[empty_corner_move-1]
        empty_side_move = np.random.choice(valid_moves)
        if empty_side_move:
            return self.moves[empty_side_move]

    def find_winning_move(self, valid_moves, board_state):
        next_move = None
        for i in range(3):
            if np.sum(board_state[i]) == 2:
                next_move = np.intersect1d(self.all_moves[i],
                                           valid_moves)[-1]
                return next_move
            elif np.sum(board_state[:, i]) == 2:
                next_move = np.intersect1d(self.all_moves[:, i],
                                           valid_moves)[-1]
                return next_move
            elif np.sum(board_state.diagonal()) == 2:
                next_move = np.intersect1d(self.all_moves.diagonal(),
                                           valid_moves)[-1]
                return next_move
            elif np.sum(np.fliplr(board_state).diagonal()) == 2:
                next_move = np.intersect1d(
                    np.fliplr(self.all_moves).diagonal(),
                    valid_moves)[-1]
                return next_move
        return next_move

    def find_forking_move(self, valid_moves, board_state):
        next_move = None
        # find forks along lines and rows
        for i in range(3):
            if (np.sum(board_state[i]) == 1 and np.sum(
                    board_state[:, np.where(board_state[i] != 1)]) == 1):
                next_move = np.intersect1d(self.all_moves[i],
                                           valid_moves)[-1] if \
                    np.intersect1d(self.all_moves[i],
                                   valid_moves).any() else None
            if next_move:
                break

        # if no forks are found try to find forks along diagonals
        if not next_move:
            if ((np.sum(board_state[0, 1:]) == 1
                 or np.sum(board_state[2, :1]) == 1)
                    and np.sum(board_state.diagonal()) == 1):
                next_move = np.intersect1d(self.all_moves.diagonal(),
                                           valid_moves)[-1] if \
                    np.intersect1d(self.all_moves.diagonal(),
                                   valid_moves).any() else None
            elif ((np.sum(board_state[0, :1]) == 1
                   or np.sum(board_state[2, 1:]) == 1)
                  and np.sum(np.fliplr(board_state).diagonal()) == 1):
                next_move = np.intersect1d(
                    np.fliplr(self.all_moves).diagonal(),
                    valid_moves)[-1] if np.intersect1d(
                        np.fliplr(self.all_moves).diagonal(),
                        valid_moves).any() else None
        return next_move

    def find_opposite_corner_move(self, valid_moves, board_state):
        next_move = None
        for i, j in itertools.combinations([0, -1], 2):
            if board_state[i, j] == -1:
                anti_i = (i+1)*-1
                anti_j = (j+1)*-1
                next_move = np.intersect1d(self.all_moves[anti_i, anti_j],
                                           valid_moves)[-1] if \
                    np.intersect1d(self.all_moves[anti_i, anti_j],
                                   valid_moves).any() else None
                break
        return next_move

    def move_manual(self, legal_moves, current_state, symbol):
        moved = False
        while not moved:
            inp = int(input('Make a move Player {}: '.format(self.name)))-1
            move = self.moves[inp]
            if move in legal_moves:
                moved = True
            else:
                print('Choose a valid move!')
        return move

    def add_state(self, encoded_state):
        pass

    def reset(self):
        pass

    def give_reward(self, reward):
        pass

    def save_training(self):
        pass


if __name__ == '__main__':
    # create Players/Agents
    a1 = Agent('Agent 1', epsilon=0.3)  # Agent 1
    a2 = Agent('Agent 2', epsilon=0.3)  # Agent 2
    p1 = Player('1')  # Human Player
    p2 = Player('Ideal 3', method='ideal')  # Ideal computer-player
    B = Board(a1, p2, p3=p2, visual=False, save=False)  # create Board
    max_rounds = 10000
    batch_size = 100
    # Train for max_rounds, test against perfect opponent after every batch
    B.alternate_train_play(batch_size, max_rounds)

    # plot state action values for some states
    state = np.zeros((3, 3), dtype=np.int64)
    a1.plot_state_action_values(state, 1, path='state_action_start.png')
    state[2, 0] = 1
    state[1, 1] = -1
    a1.plot_state_action_values(state, 1, path='state_action_1.png')
    state[2, 2] = 1
    state[0, 0] = -1
    a1.plot_state_action_values(state, 1, path='state_action_2.png')
