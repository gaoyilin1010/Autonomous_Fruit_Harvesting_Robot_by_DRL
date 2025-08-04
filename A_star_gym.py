import gym
from gym import spaces
import numpy as np
import random
import pybullet as p
import pybullet_data

# action: {0:MOVE_FORWARD, 1:MOVE_BACKWARD, 2:MOVE_Left, 3:MOVE_Right, 4:INCREASE_HEIGHT,5:DECREASE_HEIGHT}
limit_length = 1
limit_height = (4, 9)
base_size = 3
base = base_size // 2

goal_fruit = 3

class State(object):
    '''
        State.
        Implemented as 2 3d numpy arrays.
        first one "state":
            static obstacle: -1
            empty: 0
            agent = positive int(agent_id)
        second one "goals":
            agent goals = positive int(agent_id) ripe
            agent goals = negative int(agent_id) unripe
        '''

    def __init__(self, world0, goals):
        # Ensure the provided 3D arrays have the same shape and are valid
        assert (len(world0.shape) == 3 and world0.shape == goals.shape)
        # Initialize state and goal maps
        self.state = world0.copy()
        self.goals = goals.copy()

        self.agent_goals = []  # Store goal positions for the agent
        self.tree_positions = []  # Tree position and tree height (x,y,h), where h is tree height: 7
        self.agent_pos, self.agent_past, self.agent_goals, self.tree_positions = self.scanForAgent()
        self.agent_orient = 0  # The initial orientation is set to 0
        self.agent_past_orient = 0
        self.agent_d = 1  # Fixed arm length
        self.total_fruit = 0

    def scanForAgent(self):
        agent_pos = (-1, -1, -1)
        agent_past = (-1, -1, -1)
        agent_goals = []
        tree_positions = []
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                for k in range(self.state.shape[2]):
                    if (self.state[i, j, k] > 0):
                        agent_pos = (i, j, k)
                        agent_past = (i, j, k)
                    if (self.state[i, j, k - 1] == -1 and self.state[i, j, k] == 0):
                        tree_positions.append((i, j, k))
                        break
                    if (self.goals[i, j, k] != 0):
                        agent_goals.append((i, j, k))
        assert (agent_pos != (-1, -1, -1) and agent_goals != [])
        assert (agent_pos == agent_past)
        return agent_pos, agent_past, agent_goals, tree_positions

    def getPos(self):
        return self.agent_pos

    def getPastPos(self):
        return self.agent_past

    def getGoal(self):
        return self.agent_goals

    def gettree(self):
        return self.tree_positions

    def moveAgent(self, direction):
        ax, ay, az = self.agent_pos
        dx, dy, dz = direction
        x = ax + dx
        y = ay + dy
        z = az + dz

        self.state[ax, ay, az] = 0
        self.state[x, y, z] = 1
        self.agent_past = self.agent_pos
        self.agent_pos = (x, y, z)

        fruit = self.extract_fruit()

        return fruit

    # try to execture action and return whether action was executed or not and why
    # returns:
    #     2: complete goal
    #     1: action executed and reached a goal(pick a ripe fruit)
    #     0: action executed
    def act(self, action):
        # action: {0:MOVE_FORWARD, 1:MOVE_BACKWARD, 2:MOVE_RIGHT, 3:MOVE_LEFT, 4:INCREASE_HEIGHT,5:DECREASE_HEIGHT}
        direction = self.get_direction(action)
        moved = self.moveAgent(direction)
        return moved

    def get_direction(self, action):
        directions = {
            0: (1, 0, 0),
            1: (-1, 0, 0),
            2: (0, 1, 0),
            3: (0, -1, 0),
            4: (0, 0, 1),
            5: (0, 0, -1)
        }
        return directions[action]

    def extract_fruit(self):
        ax, ay, az = self.getPos()
        if (ax, ay, az) in self.agent_goals and self.goals[ax, ay, az] == 1:
            self.goals[ax, ay, az] = 0
            self.agent_goals.remove((ax, ay, az))
            self.total_fruit += 1
            if self.total_fruit >= goal_fruit:
                return 2
            return 1
        return 0


class FruitPickingEnv(gym.Env):

    def __init__(self, world0=None, goals0=None, SIZE=(10, 10, 10), tree_height=7, canopy_size=5):
        '''
        Args:
            world0, goals0: Initial world and goals. If None, they will be randomly generated.
            SIZE: Size of the 3D grid (default is 10x10x10).
            tree_height: Height of the tree trunks.
            canopy_size: Size of the tree canopy (5x5x5).
        '''
        # Initialize member variables
        self.initial_world = None
        self.initial_goals = None

        self.SIZE = SIZE
        self.tree_height = tree_height
        self.canopy_size = canopy_size
        self.fresh = True
        self.finished = False
        # Initialize data structures
        self._setWorld(world0, goals0)
        self.action_space = spaces.Discrete(6)  # 6 actions as defined
        self.viewer = None

    def _setWorld(self, world0=None, goals0=None):
        # defines the State object, which includes initializing goals and agents
        # sets the world to world0 and goals, or if they are None randomizes world
        if world0 is not None and goals0 is None:
            raise Exception("You provided a world without goals!")
        if goals0 is not None and world0 is None:
            raise Exception("You provided goals without a world!")
        # 如果提供的world0或goals0为空，就生成新的
        if world0 is None and goals0 is None:
            world, goals = self._generateTreesAndGoals()
            world = self._setAgentPosition(world)
        else:
            world, goals = world0, goals0

        self.initial_world = world.copy()  # Store the initial world state
        self.initial_goals = goals.copy()  # Store the initial goal state
        self.world = State(world, goals)  # Initialize the world with trees and goals

    def _generateTreesAndGoals(self):
        """
        Generates random trees with random fruits
        Ensures no blocked paths.
        """
        world = np.zeros(self.SIZE, dtype=int)
        goals = np.zeros(self.SIZE, dtype=int)
        col_x, row_y = (4, 4)
        # Randomize tree trunk height between 10 and 11
        trunk_height = self.tree_height
        world[col_x, row_y, :trunk_height] = -1
        # Generate random fruits in the tree canopy (5x5x5 area around the tree top)
        canopy_center = (col_x, row_y, trunk_height - 1)  # Tree top at z=trunk_height-1
        canopy = self.canopy_size // 2
        for _ in range(random.randint(4, 6)):  # Randomly generate fruits
            dx, dy, dz = random.choice([-2,2]), random.choice([-2,2]), random.choice([-2,-1,0,1,2])
            fruit_pos = (canopy_center[0] + dx, canopy_center[1] + dy, canopy_center[2] + dz)

            # Ensure the fruit position is within bounds and not on the tree trunk
            if dx == 0 and dy == 0 and dz <= 0:
                continue
            else:
                goals[fruit_pos[0], fruit_pos[1], fruit_pos[2]] = 1
        return world, goals

    # RANDOMIZE THE POSITION OF THE AGENT
    def _setAgentPosition(self, world):
        def is_valid_position(agent_x, agent_y, world):
            # Boundary check: whether the 3x3 area is outside the grid range
            if agent_x - base < 0 or agent_x + base >= self.SIZE[0] or agent_y - base < 0 or agent_y + base >= \
                    self.SIZE[1]:
                return False

            # Check if it coincides with the trunk: the z=0 height part of the 3x3 area should not have the trunk (-1)
            if np.any(world[agent_x - base:agent_x + base + 1, agent_y - base:agent_y + base + 1, 0] == -1):
                return False

            return True

        while True:
            # Randomly select agent_x, agent_y
            agent_x = np.random.randint(0, self.SIZE[0] - 1)
            agent_y = np.random.randint(0, self.SIZE[1] - 1)
            agent_h = limit_height[0]

            # Randomly select agent_x, agent_y to initialize the agent position
            if is_valid_position(agent_x, agent_y, world):
                world[agent_x, agent_y, agent_h] = 1
                break

        return world

    # Returns a 3D observation of an agent
    def _observe(self):
        obs_map = (self.world.state == -1).astype(int)
        pos_map = (self.world.state == 1).astype(int)
        goal_map = (self.world.goals == 1).astype(int)

        return [obs_map, pos_map, goal_map]

    # Resets environment

    def reset(self, world0=None, goals0=None):
        self.finished = False

        # Initialize data structures
        self._setWorld(world0, goals0)
        self.fresh = True

        if self.viewer is not None:
            self.viewer = None

        return self._observe()

    # Executes an action and returns new state, reward, done, and next valid actions
    def step(self, action):
        self.fresh = False
        n_actions = 6

        assert action in range(n_actions), 'Invalid action'

        action_status = self.world.act(action)

        if action_status == 2:
            self.finished = True  # 任务完成

        observation = self._observe()

        return observation, self.finished

    def get_joint_id(self, robot, joint_name):
        num_joints = p.getNumJoints(robot)
        for joint_id in range(num_joints):
            joint_info = p.getJointInfo(robot, joint_id)
            if joint_name == joint_info[1].decode("utf-8"):
                return joint_id
        raise ValueError(f"Joint '{joint_name}' not found.")

    def _Render(self):
        if not hasattr(self, "client_id"):
            # First GUI connection
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.setGravity(0, 0, 0)

            # Loading ground
            p.loadURDF("models/plane100.urdf", useMaximalCoordinates=True)

            # Adding a resource path
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Load the trees and fix them
            trees = self.world.gettree()
            for tree in trees:
                position = (tree[0], tree[1], 0)
                p.loadURDF("models/tree.urdf", position,
                           p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

            # Initializes the fruit store
            self.fruit_objects = []  # 用于存储所有果实对象

            # Load the initial fruit state
            goals = self.world.getGoal()
            for goal in goals:
                position = (goal[0], goal[1], goal[2])
                fruit_id = p.loadURDF("models/fruit_red.urdf", position,
                                      p.getQuaternionFromEuler([0, 0, 0]))
                self.fruit_objects.append(fruit_id)

            # Load the robot model and fix it
            agent_pos = self.world.getPos()
            self.robot = p.loadURDF("models/robot_astar.urdf", agent_pos,
                                    p.getQuaternionFromEuler([0, 0, 0]), globalScaling=10.0)

            # Enable render
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            # Set up camera
            cameraDistance = 15
            cameraYaw = 30
            cameraPitch = -20
            cameraTargetPosition = agent_pos
            p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # Renew fruit status
        goals = self.world.getGoal()
        # Remove old fruit
        for fruit_id in self.fruit_objects:
            p.removeBody(fruit_id)
        self.fruit_objects = []

        # Load new fruits state
        for goal in goals:
            position = (goal[0], goal[1], goal[2])
            fruit_id = p.loadURDF("models/fruit_red.urdf", position, p.getQuaternionFromEuler([0, 0, 0]))
            self.fruit_objects.append(fruit_id)

        # Update robot position and joint status
        x, y, h = self.world.getPos()
        h = min(h - 4, 4)

        p.resetBasePositionAndOrientation(self.robot, [x, y, 0], p.getQuaternionFromEuler([0, 0, 0]))
        vertical_extend_joint = self.get_joint_id(self.robot, "vertical_extend_joint")
        p.resetJointState(self.robot, vertical_extend_joint, h)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.stepSimulation()
