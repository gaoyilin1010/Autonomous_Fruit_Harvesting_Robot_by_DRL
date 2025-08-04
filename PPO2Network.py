import torch
import torch.nn as nn
import torch.nn.functional as F

'''
inputs: pos_map, goal_map, obs_map
Channel 1: The robot's position and orientation in the environment.
Channel 2: The position and ripeness of the fruit.
Channel 3: Obstacle or boundary information.
'''

# NN Parameters
net_size = 128

class PPO2Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPO2Network, self).__init__()

        # Inputs size
        self.state_size = state_size  # 1x3x10x10x10, as input
        self.action_size = action_size

        # Residual block
        self.shortcut1 = nn.Conv3d(in_channels=state_size[1], out_channels=net_size//2, kernel_size=2, stride = 2) # 64x5x5x5
        self.shortcut2 = nn.Conv3d(in_channels=net_size//2, out_channels=net_size, kernel_size=5, stride = 5)# 128x1x1x1

        # Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=state_size[1], out_channels=net_size//2, kernel_size=3, stride=1, padding=1)  # 64x10x10x10
        self.conv1a = nn.Conv3d(in_channels=net_size//2, out_channels=net_size//2, kernel_size=3, stride=1, padding=1) # 64x10x10x10
        self.conv2 = nn.Conv3d(in_channels=net_size//2, out_channels=net_size, kernel_size=3, stride=1, padding=1) # 128x5x5x5
        self.conv2a = nn.Conv3d(in_channels=net_size, out_channels=net_size, kernel_size=3, stride=1, padding=1) # 128x5x5x5

        # BatchNorm3d
        self.bn1 = nn.BatchNorm3d(net_size//2)
        self.bn2 = nn.BatchNorm3d(net_size)

        # Actication function
        self.relu = nn.ReLU()

        # Max-pooling
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64x5x5x5
        self.maxpool2 = nn.MaxPool3d(kernel_size=5, stride=5)  # 128x1x1x1

        # Fully-connected layers
        self.fc1 = nn.Linear(net_size, net_size)  # 128
        self.fc2 = nn.Linear(net_size, net_size)  # 128

        # The policy and value network
        self.policy = nn.Linear(net_size, action_size)
        self.value = nn.Linear(net_size, 1)


    def forward(self, inputs, lstm_hidden_state=None):
        '''Resnet Part'''
        # conv1 -> conv1a -> maxpool + residual connection
        shortcut1 = self.shortcut1(inputs)  # 64x5x5x5
        x = self.bn1(self.conv1(inputs))  # 64x10x10x10
        x = self.relu(x)
        x = self.bn1(self.conv1a(x))  # 64x10x10x10
        x = self.relu(x)
        x = self.maxpool1(x)  # 64x5x5x5
        x = self.relu(x + shortcut1)  # 64x5x5x5

        # conv2 -> conv2a -> maxpool + residual connection
        shortcut2 = self.shortcut2(x)  # 128x1x1x1
        x = self.bn2(self.conv2(x))  # 128x5x5x5
        x = self.relu(x)
        x = self.bn2(self.conv2a(x)) # 128x5x5x5
        x = self.relu(x)
        x = self.maxpool2(x)  # 128x1x1x1
        x = self.relu(x + shortcut2)  # 128x1x1x1

        # Fully-connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))  # 128x1
        x = self.relu(self.fc2(x))  # 128x1

        '''Policy and Value Part'''
        policy = F.softplus(self.policy(x))  # 7x1, softplus activation ensures output > 0
        value = self.value(x)  # 1x1, scalar

        mean = torch.tanh(self.policy(x))  # Output probability distribution mean
        std = policy
        Q = value

        return mean, std, Q