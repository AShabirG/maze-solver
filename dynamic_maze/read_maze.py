import os

import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(2022)

flag_list = [0, 1, 2, 3, 5, 6, 7, 8]
#  [0, 1, 2
#   3,    5
#   6, 7, 8]

time_list = [0, 1, 2]

# the maze of size 201*201*2
maze_cells = np.zeros((201, 201, 2), dtype=int)

# load maze
def load_maze():
    if not os.path.exists("COMP6247Maze20212022.npy"):
        raise ValueError("Cannot find %s" % file_path)

    else:
        global maze_cells
        maze = np.load("COMP6247Maze20212022.npy", allow_pickle=False, fix_imports=True)
        maze_cells = np.zeros((maze.shape[0], maze.shape[1], 2), dtype=int)
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                maze_cells[i][j][0] = maze[i][j]
                # load the maze, with 1 denoting an empty location and 0 denoting a wall
                maze_cells[i][j][1] = 0
                # initialized to 0 denoting no fire

# get local 3*3 information centered at (x,y).
def get_local_maze_information(x, y):
    global maze_cells
    random_location = random.choice(flag_list)
    around = np.zeros((3, 3, 2), dtype=int)
    for i in range(maze_cells.shape[0]):
        for j in range(maze_cells.shape[1]):
            if maze_cells[i][j][1] == 0:
                pass
            else:
                maze_cells[i][j][1] = maze_cells[i][j][1] - 1  # decrement the fire time

    for i in range(3):
        for j in range(3):
            if x - 1 + i < 0 or x - 1 + i >= maze_cells.shape[0] or y - 1 + j < 0 or y - 1 + j >= maze_cells.shape[1]:
                around[i][j][0] = 0  # this cell is outside the maze, and we set it to a wall
                around[i][j][1] = 0
                continue
            around[i][j][0] = maze_cells[x - 1 + i][y - 1 + j][0]
            around[i][j][1] = maze_cells[x - 1 + i][y - 1 + j][1]
            if i == random_location // 3 and j == random_location % 3:
                if around[i][j][0] == 0: # this cell is a wall
                    continue
                ran_time = random.choice(time_list)
                around[i][j][1] = ran_time + around[i][j][1]
                maze_cells[x - 1 + i][y - 1 + j][1] = around[i][j][1]
    return around
# def get_local_maze_information(x,y):
#     global maze_cells
#     random_location = random.choice(flag_list)
#     around = np.zeros((3, 3, 2), dtype=int)
#     xval=2
#     yval=2
#     if x<2 or x>199:
#         xval=1
#     if y<2 or y>199:
#         yval=1
#     for row in range(x-xval,x+xval):
#         for col in range(y-yval,y+yval):
#             if maze_cells[row][col][1] == 0:
#                 pass
#             else:
#                 maze_cells[row][col][1] -= 1
#
#     for row in range(3):
#         for col in range(3):
#             if x - 1 + row < 0 or x - 1 + row >= maze_cells.shape[0] or y - 1 + col < 0 or y -1 + col >= maze_cells.shape[1]:
#                 around[row][col][0] = 0
#                 around[row][col][1] = 0
#                 continue
#             around[row][col][0] = maze_cells[x - 1 + row][y - 1 + col][0]
#             around[row][col][1] = maze_cells[x - 1 + row][y - 1 + col][1]
#             if row == random_location // 3 and col == random_location % 3:
#                 if around[row][col][0] == 0: # this cell is a wall
#                     continue
#                 ran_time = random.choice(time_list)
#                 around[row][col][1] = ran_time + around[row][col][1]
#                 maze_cells[x - 1 + row][y - 1 + col][1] = around[row][col][1]
#     return around
# load_maze()
# k = maze_cells
# maze = k[:,:5, 0]
# plt.imshow(maze)
# plt.show()
# print("hi")
#
# if __name__ == '__main__':
#     load_maze()
#     k= maze_cells
#     point = (5, 7)
#     load_maze()
#     vicinity = get_local_maze_information(*point)
#     print(f"vicinity: {vicinity}")
#
#
#     is_direction = lambda idx: idx % 2 == 0
#     directions = [val for idx, val in enumerate(vicinity.flatten()) if is_direction(idx)]
#
#     from enum import Flag, auto, Enum
#
#     class Direction(Flag):
#         wall = auto()
#         free = auto()
#
    # class Actions(Enum):
    #     UP_LEFT = 0
    #     UP = 1
    #     UP_RIGHT = 2
    #     LEFT = 3
    #     CENTER = 4
    #     RIGHT = 5
    #     DOWN_LEFT = 6
    #     DOWN = 7
    #     DOWN_RIGHT = 8
#
#     viable = lambda direction_array: [Actions(idx) for idx, val in enumerate(direction_array) if Direction(val) == Direction.wall]
#
#     reward_func = lambda count: -0.05*count + euclcidean_distance_from_start() + (euclidean distance history) +

