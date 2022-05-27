from __future__ import print_function

import operator
import random
import read_maze as rm
from enum import IntEnum
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# All possible actions for the actor in an enumerated class. Enum based on position given by local maze information func
class Actions(IntEnum):
    UP = 1
    LEFT = 3
    STAY = 4
    RIGHT = 5
    DOWN = 7


# Load maze, purely for visual purposes, not used by algorithm
maze = rm.load_maze()
k = rm.maze_cells


# Two states for whether there is or isn't a wall
class Direction(IntEnum):
    wall = 0
    free = 1


# Two states fire or not, length does not matter
class FirePresent(IntEnum):
    no_fire = 0
    fire = 1


class MazeSolver:
    # q table dimensions are x pos, y pos and action
    # This dictionary maps actions to integers for the depth of the q table dimension
    q_table_action_mapping = {0: Actions.UP, 1: Actions.LEFT, 2: Actions.STAY, 3: Actions.RIGHT, 4: Actions.DOWN}
    # Reverse mapping of above
    q_table_reverse_mapping = {Actions.UP: 0, Actions.LEFT: 1, Actions.STAY: 2, Actions.RIGHT: 3, Actions.DOWN: 4}
    # Opposite direction of action for all actions. Used as reward
    reverse_direction = {Actions.UP: Actions.DOWN, Actions.LEFT: Actions.RIGHT, Actions.RIGHT: Actions.LEFT,
                         Actions.DOWN: Actions.UP, Actions.STAY: Actions.STAY}

    def __init__(self):
        print("Started.")
        self.step_count = 0
        # Load maze for drawing only
        self.maze = k[:, :, 0]
        # Load fires for drawing only
        self.fires = k[:, :, 1]
        # Exploration factor
        self.epsilon = 0.1

        # Various rewards
        self.wall_penalty = -1.0
        self.win_reward = 1.0
        self.lose_reward = -1.0
        self.step_reward = -0.1
        self.discount_factor = 0.5
        self.fire_reward = -1.0
        self.visited_reward = -0.2
        self.unvisited_reward = 0.2
        self.total_reward = 0

        self.learning_rate = 0.1
        # Empty Q table initialisation
        self.q_table = np.zeros((200, 200, 5))
        # Checking if actor stands on fire
        self.is_onfire = False
        self.fires_stepped_on = 0
        # Initiate Epoch value
        self.epoch = 1
        self.step_count_per_epoch = []
        # Initialise actor position
        self.actor = [1, 1]
        # Start path
        self.path = np.ones((1, 2))
        # Dictionary for insuring repeat values are not added to path
        self.position_history = defaultdict(lambda: 0)

    def vicinity(self):
        # Get local maze information
        return rm.get_local_maze_information(self.actor[1], self.actor[0])

    def reset(self):
        # Reset required variables
        self.actor = [1, 1]
        self.path = np.ones((1, 2))
        self.total_reward = 0
        self.step_count = 0
        self.position_history = defaultdict(lambda: 0)
        self.epoch += 1
        # End if too many epochs have elapsed and return best q table epoch.
        global go
        if self.epoch >= 21:
            go = False
            best_run = min(self.step_count_per_epoch)
            index_of_best = self.step_count_per_epoch.index(best_run)
            print(f"Best q table was produced in epoch {index_of_best + 1}")

    def step(self):
        """Run the pygame environment for displaying the maze structure and visible (local) environment of actor
        """
        # Check if game is over
        self.check_terminal_state()
        # Check legal actions in current position
        viable, fires = self.viable_actions()
        # Determine actions using q table and the legal actions (these represent the state)
        self.action(viable, fires)
        # Increment the step count
        self.step_count += 1

    def viable_actions(self):
        """
        Function calling to get local maze information. This information is then used to determine which of the 5
        actions are legal through checking if it may result in walking into a wall or fire. These actions are then
        removed from the candidate moves for that state and a choice is made from the remaining moves later in the code.
        :return:
        Returns the legal moves available in the state in the form described in the Actions class.
        """

        # Call local maze information function
        vicinity = self.vicinity()

        # When vicinity is flattened every second element corresponds to either free path or wall. The remainder denote
        # state of fire
        is_direction = lambda idx: idx % 2 == 0
        directions_available = [val for idx, val in enumerate(vicinity.flatten()) if is_direction(idx)]

        # Opposite of above line. Only interested in the state of the fires.
        is_fire = lambda idx: idx % 2 != 0
        loc_fire = [val for idx, val in enumerate(vicinity.flatten()) if is_fire(idx)]
        # Length of fire does not matter so if the fire is greater than one, just set it to one for enumeration purposes
        for i in range(len(loc_fire)):
            if loc_fire[i] != 0:
                loc_fire[i] = 1

        # Check for if action will run into a wall and if it does to remove it from the list of available actions.
        # The returned viable_actions variable contains the legal actions in the position based on walls
        viable = lambda direction_array: [Actions(idx) for idx, val in enumerate(direction_array) if
                                          Direction(val) != Direction.wall and idx in Actions._value2member_map_]
        viable_actions = viable(directions_available)

        # Check for if the action would lead to walking into a fire. If it does add it to variable "fires"
        fire = lambda direction_array: [(Actions(idx)) for idx, val in enumerate(direction_array) if
                                        Direction(val) == FirePresent.fire and idx in Actions._value2member_map_]
        fires = fire(loc_fire)

        # Remove any actions that result into walking into fire from the list of viable actions
        for i in fires:
            if i in viable_actions:
                viable_actions.remove(i)
        return viable_actions, fires

    def action(self, viable_actions, fires):

        # Select a random action if below assigned exploration value
        if random.random() < self.epsilon:
            action = viable_actions[random.randint(0, len(viable_actions) - 1)]

        # Otherwise, Select the action with the highest q value
        else:
            # Load all q values of the 5 moves in the current state into a list 'options'
            options = self.q_table[self.actor[1], self.actor[0], :]

            # Remove the q values that correspond to a move which will lead into a illegal position as determined by
            # viable actions
            viable_options = {MazeSolver.q_table_action_mapping[idx]: val for idx, val in enumerate(options) if
                              MazeSolver.q_table_action_mapping[idx] in viable_actions}

            # Choose the action which has the best q value of the remaining legal actions
            action = max(viable_options.items(), key=operator.itemgetter(1))[0]

        # Check if action results in standing on a fire
        self.is_onfire = action in fires
        if self.is_onfire:
            self.fires_stepped_on += 1

        # Copy old actor for Bellmans equation
        old_actor = np.copy(self.actor)
        # Apply action
        if action == Actions.UP:
            self.up()
        elif action == Actions.LEFT:
            self.left()
        elif action == Actions.DOWN:
            self.down()
        elif action == Actions.RIGHT:
            self.right()
        elif action == Actions.STAY:
            self.stay()
        else:
            print("Invalid action attempted.")

        # If the resultant state has not been seen before add it to the path
        if (self.actor[0] + 200 * self.actor[1]) not in self.position_history:
            add_to_path = np.empty((1, 2))
            add_to_path[0][0] = self.actor[1]
            add_to_path[0][1] = self.actor[0]
            self.path = np.append(self.path, add_to_path, axis=0)

        # Inform user if a fire was stepped on
        if self.is_onfire:
            print(f"Stepped on fire on step {self.step_count}")

        # Add to position history
        self.position_history[self.position_index_1D()] += 1

        # Calculate reward of move
        resultant_reward = self.reward_check()
        # Determine max q value for new state for Bellmans
        max_q = self.q_table[self.actor[1], self.actor[0], :].max()

        # Assign q value to the q table using Bellmans equation
        self.q_table[old_actor[1], old_actor[0], MazeSolver.q_table_reverse_mapping[action]] += self.learning_rate * \
            (resultant_reward + self.discount_factor * max_q - self.q_table[old_actor[1], old_actor[0],
                                                                            MazeSolver.q_table_reverse_mapping[action]])

        # For the new state add a negative reward for the reverse direction of the current action to penalise going back
        self.q_table[self.actor[1], self.actor[0], MazeSolver.q_table_reverse_mapping[
            MazeSolver.reverse_direction[action]]] += self.learning_rate * self.visited_reward

        # Total reward thus far
        self.total_reward += resultant_reward

    def position_index_1D(self):
        # Convert position history to 1 dimension to save memory and ensure they don't overlap using multiplier.
        return self.actor[0] + self.actor[1] * 200

    # Actions and their change to position of the actor
    def up(self):
        self.actor[1] -= 1

    def down(self):
        self.actor[1] += 1

    def right(self):
        self.actor[0] += 1

    def left(self):
        self.actor[0] -= 1

    def stay(self):
        pass

    # Check reward for the action just carried out
    def reward_check(self):
        reward = self.step_reward + int(self.is_onfire) * self.fire_reward

        # Win reward condition
        if self.actor[0] == 199 and self.actor[1] == 199:
            reward += self.win_reward

        # If an old position is revisited penalise the action
        if (self.actor[0] + 200 * self.actor[1]) in self.position_history:
            reward += self.visited_reward

        # Experimental reward for actions that lead to a new state
        # if (self.actor[0] + 200 * self.actor[1]) not in self.position_history:
        #     reward += self.unvisited_reward

        # Return total reward for the action
        return reward

    def graphics_matplot(self):
        # Plot the maze by first expanding it to span 0 to 255
        maze_print = np.copy(self.maze) * 255

        # Plot the path of the actor through the maze with half the grayscale value
        for i in range(self.path.shape[0]):
            maze_print[int(self.path[i][0])][int(self.path[i][1])] = 0.5 * 255

        # Print the maze and path and save it
        plt.imshow(maze_print)
        plt.savefig(f"epoch{self.epoch}.png")

    def check_terminal_state(self):
        # Check whether to end the game
        if self.actor[0] == 199 and self.actor[1] == 199:
            print("Exit reached")
            print(f'It took {self.step_count} steps.')
            print(f'Epoch:{self.epoch}')
            print(f'Fires stepped on: {self.fires_stepped_on}')
            self.save_q_table()
            self.step_count_per_epoch.append(self.step_count)
            self.graphics_matplot()
            self.reset()

        # Progress report
        elif self.step_count % 10000 == 0:
            print(f'{self.step_count} steps.')
            print(f'Fires stepped on: {self.fires_stepped_on}')
            self.save_q_table()
            self.step_count_per_epoch.append(self.step_count)
            self.graphics_matplot()

        # Limit steps per epoch before restart
        if self.step_count > 5000000:
            self.graphics_matplot()
            print("Exceeded maximum step count of 10,000,000 for a single epoch; actor position has been reset to "
                  "start point.")
            self.reset()

    # Save q table to npy file
    def save_q_table(self):
        np.save(f"Epoch{self.epoch}", self.q_table)


go = True
screen = MazeSolver()

while go:
    # User must stop as it will keep going and slightly improve over time. Reducing number of steps.
    screen.step()
