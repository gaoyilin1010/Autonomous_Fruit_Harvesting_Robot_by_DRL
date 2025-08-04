import numpy as np
import heapq


def a_star(start, goal, obstacles):
    '''
    A* algorithm: two-dimensional grid path planning (only support up, down, left and right four directions).
    '''
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left and right
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # Loop until open_set is empty (the search space has been fully explored or the target point found)
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for dx, dy in directions:  # Traverses all adjacent nodes of the current node
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < obstacles.shape[0] and 0 <= neighbor[1] < obstacles.shape[1]):
                # Check whether neighboring nodes are within the map range
                continue
            if obstacles[neighbor[0], neighbor[1]] == 1:
                # Check whether adjacent nodes are obstructions
                continue
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []


def heuristic(a, b):
    '''Heuristic function: Manhattan distance'''
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from, current):
    '''Rebuild the path according to came_from'''
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# Extract fruit section: for robotic arm
def move_arm_to_height(current_h, target_h):
    '''
    Robot arm vertical expansion logic.
    Parameters:
    - current_h: specifies the height of the current robot arm.
    - target_h: height of the target fruit.
    Back:
    - Action sequences (e.g. ["up", "up", "down"]).
    '''

    actions = []
    while current_h < target_h:
        actions.append("up")
        current_h += 1
    while current_h > target_h:
        actions.append("down")
        current_h -= 1
    return actions
