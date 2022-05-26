from __future__ import print_function

import operator
import random
import time
from PIL import Image
import pygame
import sys
import matplotlib.pyplot as plt
import read_maze as rm
from enum import Flag, auto, Enum, IntEnum
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
SCREENSIZE = W, H = 600, 600
mazeWH = 600
origin = ((W - mazeWH) / 2, (H - mazeWH) / 2)
lw = 2  # linewidth of maze-grid

# Colours
GREY = (140, 140, 140)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (200, 208, 16)

class Actions(IntEnum):
    UP = 1
    LEFT = 3
    STAY = 4
    RIGHT = 5
    DOWN = 7


num_actions = len(Actions)
maze = rm.load_maze()
k = rm.maze_cells


class Direction(IntEnum):
    wall = 0
    free = 1


class FirePresent(IntEnum):
    no_fire = 0
    fire = 1


class Screen:
    q_table_action_mapping = {0: Actions.UP, 1: Actions.LEFT, 2: Actions.STAY, 3: Actions.RIGHT, 4: Actions.DOWN}
    q_table_reverse_mapping = {Actions.UP: 0, Actions.LEFT: 1, Actions.STAY: 2, Actions.RIGHT: 3, Actions.DOWN: 4}
    reverse_direction = {Actions.UP: Actions.DOWN, Actions.LEFT: Actions.RIGHT, Actions.RIGHT: Actions.LEFT,
                         Actions.DOWN: Actions.UP, Actions.STAY: Actions.STAY}

    def __init__(self):
        self.step_count = 0
        self.maze = k[:, :, 0]
        self.fires = k[:, :, 1]
        self.epsilon = 0.1
        self.shape = 201
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
        self.q_table = np.zeros((200, 200, 5))
        self.is_onfire = False
        self.epoch = 1
        pygame.init()

        #self.surface = pygame.display.set_mode(SCREENSIZE)
        self.actor = [1, 1]
        self.path = np.ones((1, 2))
        self.position_history = defaultdict(lambda: 0)
        # self.graphics_step()

    def drawSquareCell(self, x, y, dimX, dimY, col=(0, 0, 0)):
        pygame.draw.rect(
            self.surface, col,
            (y, x, dimX, dimY)
        )

    def vicinity(self):
        return rm.get_local_maze_information(self.actor[1], self.actor[0])

    def reset(self):
        self.actor = [1, 1]
        self.path = np.ones((1, 2))
        self.total_reward = 0
        self.step_count = 0
        self.position_history = defaultdict(lambda: 0)
        self.get_event()
        self.epoch += 1
        pygame.init()

    def drawSquareGrid(self, origin, gridWH):
        CONTAINER_WIDTH_HEIGHT = gridWH
        cont_x, cont_y = origin

        # DRAW Grid Border:
        # TOP lEFT TO RIGHT
        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y), lw)
        # # BOTTOM lEFT TO RIGHT
        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, CONTAINER_WIDTH_HEIGHT + cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x,
             CONTAINER_WIDTH_HEIGHT + cont_y), lw)
        # # LEFT TOP TO BOTTOM
        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, cont_y),
            (cont_x, cont_y + CONTAINER_WIDTH_HEIGHT), lw)
        # # RIGHT TOP TO BOTTOM
        pygame.draw.line(
            self.surface, BLACK,
            (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x,
             CONTAINER_WIDTH_HEIGHT + cont_y), lw)

    def placeCells(self):
        # GET CELL DIMENSIONS...
        celldimX = celldimY = (mazeWH / self.shape)

        for rows in range(201):
            for cols in range(201):
                if self.maze[rows][cols] == 0:
                    self.drawSquareCell(
                        origin[0] + (celldimY * rows) + lw / 2,
                        origin[1] + (celldimX * cols)
                        + lw / 2,
                        celldimX, celldimY, col=BLACK)

                elif self.fires[rows][cols] == 1:
                    self.drawSquareCell(
                        origin[0] + (celldimY * rows)
                        + lw / 2,
                        origin[1] + (celldimX * cols)
                        + lw / 2,
                        celldimX, celldimY, col=RED)
                # Draw end point
                elif cols == 199 and rows == 199:
                    self.drawSquareCell(
                        origin[0] + (celldimY * rows)
                        + lw / 2,
                        origin[1] + (celldimX * cols)
                        + lw / 2,
                        celldimX, celldimY, col=BLUE)

    def drawPath(self):
        celldimX = celldimY = (mazeWH / self.shape)
        for rows in range(self.path.shape[0]):
            self.drawSquareCell(
                origin[0] + (celldimY * self.path[rows][0])
                + lw / 2,
                origin[1] + (celldimX * self.path[rows][1])
                + lw / 2,
                celldimX, celldimY, col=YELLOW)

    def drawActor(self):
        celldimX = celldimY = (mazeWH / self.shape)
        self.drawSquareCell(
            origin[0] + (celldimY * self.actor[1])
            + lw / 2,
            origin[1] + (celldimX * self.actor[0])
            + lw / 2,
            celldimX, celldimY, col=BLUE)

    def step(self):
        """Run the pygame environment for displaying the maze structure and visible (local) environment of actor
        """
        self.check_terminal_state()
        viable, fires = self.viable_actions()
        self.action(viable, fires)
        self.step_count += 1
        #self.graphics_step()

    def viable_actions(self):
        vicinity = self.vicinity()
        is_direction = lambda idx: idx % 2 == 0
        directions_available = [val for idx, val in enumerate(vicinity.flatten()) if is_direction(idx)]
        is_fire = lambda idx: idx % 2 != 0
        loc_fire = [val for idx, val in enumerate(vicinity.flatten()) if is_fire(idx)]
        for i in range(len(loc_fire)):
            if loc_fire[i] != 0:
                loc_fire[i] = 1
        viable = lambda direction_array: [Actions(idx) for idx, val in enumerate(direction_array) if
                                          Direction(val) != Direction.wall and idx in Actions._value2member_map_]
        viable_actions = viable(directions_available)
        fire = lambda direction_array: [(Actions(idx)) for idx, val in enumerate(direction_array) if
                                        Direction(val) == FirePresent.fire and idx in Actions._value2member_map_]
        fires = fire(loc_fire)

        for i in fires:
            if i in viable_actions:
                viable_actions.remove(i)
        return viable_actions, fires

    def action(self, viable_actions, fires):
        # Select a random action
        if random.random() < self.epsilon:
            action = viable_actions[random.randint(0, len(viable_actions) - 1)]

        # Select the action with the highest q
        else:

            options = self.q_table[self.actor[1], self.actor[0], :]
            viable_options = {Screen.q_table_action_mapping[idx]: val for idx, val in enumerate(options) if
                              Screen.q_table_action_mapping[idx] in viable_actions}
            action = max(viable_options.items(), key=operator.itemgetter(1))[0]

        self.is_onfire = action in fires
        old_actor = np.copy(self.actor)
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
        if (self.actor[0] + 200 * self.actor[1]) not in self.position_history:
            add_to_path = np.empty((1, 2))
            add_to_path[0][0] = self.actor[1]
            add_to_path[0][1] = self.actor[0]
            self.path = np.append(self.path, add_to_path, axis=0)

        if self.is_onfire:
            print(f"Stepped on fire on step {self.step_count}")

        self.position_history[self.position_index_1D()] += 1
        resultant_reward = self.reward_check()
        max_q = self.q_table[self.actor[1], self.actor[0], :].max()
        self.q_table[old_actor[1], old_actor[0], Screen.q_table_reverse_mapping[action]] += self.learning_rate * \
                (resultant_reward + self.discount_factor * max_q - self.q_table[old_actor[1], old_actor[0],
                                                                                Screen.q_table_reverse_mapping[action]])
        self.q_table[self.actor[1], self.actor[0], Screen.q_table_reverse_mapping[
            Screen.reverse_direction[action]]] += self.learning_rate * self.visited_reward
        self.total_reward += resultant_reward
        return action

    def position_index_1D(self):
        return self.actor[0] + self.actor[1] * 200

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

    def reward_check(self):
        reward = self.step_reward + int(self.is_onfire) * self.fire_reward

        if self.actor[0] == 199 and self.actor[1] == 199:
            reward += self.win_reward
        if (self.actor[0] + 200 * self.actor[1]) in self.position_history:
            reward += self.visited_reward
        # if (self.actor[0] + 200 * self.actor[1]) not in self.position_history:
        #     reward += self.unvisited_reward

        return reward

    def graphics_step(self):
        self.get_event()
        self.surface.fill(GREY)
        self.drawSquareGrid(origin, mazeWH)
        self.drawPath()
        self.drawActor()
        self.placeCells()
        pygame.display.update()

    def graphics_matplot(self):
        horse = np.copy(self.maze) *255
        for i in range(self.path.shape[0]):
            horse[int(self.path[i][0])][int(self.path[i][1])] = 0.5 * 255
        #k = Image.fromarray(horse)
        plt.imshow(horse)
        plt.savefig(f"epoch{self.epoch}.png")

    def check_terminal_state(self):
        if self.actor[0] == 199 and self.actor[1] == 199:
            print("Exit reached")
            print(f'It took {self.step_count} steps.')
            print(f'Epoch:{self.epoch}')
            print(f'Fires stepped on: {self.fires_stepped_on}')
            #self.graphics_step()
            self.graphics_matplot()
            #time.sleep(5)
            self.reset()
        if self.step_count % 100000 == 0:
            #self.graphics_step()
            self.graphics_matplot()
            print(f'{self.step_count} steps.')
        if self.step_count > 10000000:
            self.graphics_matplot()
            #self.graphics_step()
            print("failed")
            #time.sleep(5)
            self.reset()# else self.reward
        # Else statement required for if taking too long and reward drops below threshold

    def get_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()


screen = Screen()
while 1:
    screen.step()
