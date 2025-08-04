import random
import time
import pybullet as p
import pybullet_data
import numpy as np
from A_star_gym import FruitPickingEnv
from A_star import a_star


def extract_environment_data(obs_map, pos_map, goal_map):
    '''
    Extract starting point, target point, and obstacle maps (for 3D discrete maps).
    Parameters:
    obs_map: obstacle map (3D numpy array, 0: traffic, 1: obstacle)
    - pos_map: robot location map (3D numpy array, 1 indicates the current location of the robot)
    - goal_map: target point map (3D numpy array, 1 indicates the location of the fruit)
    Back:
    -start: start position of the robot (x, y, z)
    - goals: list of goals [(x1, y1, z1), (x2, y2, z2),...]
    - obstacles: two-dimensional plane list of obstacles [(x1, y1), (x2, y2),...]
    '''

    # Find the starting position of the robot (x, y, z)
    start = tuple(np.argwhere(pos_map == 1)[0])

    # Find all target points [(x1, y1, z1), (x2, y2, z2),...]
    goals = [tuple(pos) for pos in np.argwhere(goal_map == 1)]

    # Extract two-dimensional plane coordinates of obstacles (layer z = 0)
    obstacles = obs_map[:, :, 0]

    # Traverse the location of the obstacle and set its neighborhood as the obstacle
    for x, y in zip(*np.where(obstacles == 1)):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                obstacles[x + dx, y + dy] = 1

    print('start: ', start)
    print('goals: ', goals)

    return start, goals, obstacles


def get_action_from_coordinates(current_pos, next_pos):
    # Calculate the coordinate difference
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]

    # Map actions according to the difference
    if dx == 1 and dy == 0:
        action = 0  # Up
    elif dx == -1 and dy == 0:
        action = 1  # Down
    elif dx == 0 and dy == 1:
        action = 2  # Left
    elif dx == 0 and dy == -1:
        action = 3  # Right
    else:
        raise ValueError(f"Invalid coordinate change: dx={dx}, dy={dy}")

    return action


if __name__ == '__main__':
    # Initialize the environment
    env = FruitPickingEnv()
    # First render environment
    env._Render()
    obs_map, pos_map, goal_map = env._observe()
    actions = []
    # Extract environmental data
    start, goals, obstacles = extract_environment_data(obs_map, pos_map, goal_map)

    # Initial height of the current robotic arm
    current_h = start[2]
    # Traversal each fruit
    for goal in goals:
        print(f"Plan the path to the target fruit: {goal}")
        # Extract the height of the target fruit
        target_h = goal[2]

        # A* Path planning
        path = a_star(start[:2], goal[:2], obstacles)
        print(f"Planned path: {path}")

        # Initial position of current robot
        current_pos = start[:2]

        # Move the robot along the path
        for step in path[1:]:
            # Get action from coordinates
            action = get_action_from_coordinates(current_pos, step)
            actions.append(action)
            current_pos = step

        # Adjust the height of the robot arm to pick fruit
        arm_actions = target_h - current_h
        if arm_actions > 0:
            for action in range(arm_actions):
                actions.append(4)
        elif arm_actions < 0:
            for action in range(-arm_actions):
                actions.append(5)
        current_h = target_h  # Updated robotic arm height
        start = current_pos

    print(actions)

    for action in actions:
        time.sleep(0.5)  # Perform an action every 0.5 seconds
        env.step(action)  # Perform actions
        env._Render()  # Render new state

    # Enter a continuous render loop and keep the window open
    while True:
        p.stepSimulation()  # Run physics simulations continuously
        time.sleep(1. / 10.)  # Control simulation step frequency
