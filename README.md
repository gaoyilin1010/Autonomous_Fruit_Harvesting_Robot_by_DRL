# Autonomous_Fruit_Harvesting_Robot_by_DRL
ME5418 Project

## Overview
This section implements a RL agent based on the PPO2 algorithm. 
The goals of this project are to train a fruit-picking robot by PPO2 algorithm to learn how to perform effective picking operations in a three-dimensional discrete space and to show a rendered video of the robot executing the policy.

## File Structure
├── A* &nbsp;&nbsp;&nbsp;&nbsp;  
│   └── A_star_gym.py &nbsp;&nbsp;&nbsp;&nbsp; # A* environment  
│   └── A_star.py &nbsp;&nbsp;&nbsp;&nbsp; # A* function  
│   └── A_star_agent.py &nbsp;&nbsp;&nbsp;&nbsp; # A* implementation  
├── PPO2 &nbsp;&nbsp;&nbsp;&nbsp;   
│   └── our_gym.py &nbsp;&nbsp;&nbsp;&nbsp; # Our project environment  
│   └── PPO2Network.py &nbsp;&nbsp;&nbsp;&nbsp; # Defines the PPO2Network neural network architecture   
│   └── PPO2Agent.py &nbsp;&nbsp;&nbsp;&nbsp; # Defines the PPO2Agent actor and critic architecture with ppo2train and ppo2test processes  
├── PPO2_easy &nbsp;&nbsp;&nbsp;&nbsp;   
│   └── our_gym_easy.py &nbsp;&nbsp;&nbsp;&nbsp; # Our project environment simplified for testing  
│   └── PPO2Network.py &nbsp;&nbsp;&nbsp;&nbsp; # the same file as above (in PPO2 part)  
│   └── PPO2Agent_easy.py &nbsp;&nbsp;&nbsp;&nbsp; # Defines the PPO2Agent actor and critic architecture with ppo2train and ppo2test processes simplified for testing  
├── Models &nbsp;&nbsp;&nbsp;&nbsp; # Our urdf files of models  
│   └── fruit_red.urdf &nbsp;&nbsp;&nbsp;&nbsp; # Model of ripe fruit  
│   └── fruit_yellow.urdf &nbsp;&nbsp;&nbsp;&nbsp; # Model of immature fruit  
│   └── plane100.urdf &nbsp;&nbsp;&nbsp;&nbsp; # Model of ground  
│   └── plane100.obj &nbsp;&nbsp;&nbsp;&nbsp; # Model of ground (obj)  
│   └── robot.urdf &nbsp;&nbsp;&nbsp;&nbsp; # Model of robot  
│   └── tree.urdf &nbsp;&nbsp;&nbsp;&nbsp; # Model of tree  
│   └── robot_astar.urdf &nbsp;&nbsp;&nbsp;&nbsp; # Model of robot for A*  
├── PPO2_Models &nbsp;&nbsp;&nbsp;&nbsp; # Store actor.pth and critic.pth files  
│   └── PPO2_model_actor.pth &nbsp;&nbsp;&nbsp;&nbsp; # Stores the weight of actor network in PPO2Agent  
│   └── PPO2_model_critic.pth &nbsp;&nbsp;&nbsp;&nbsp; # Stores the weight of critic network in PPO2Agent  
│   └── PPO2_model_actor_easy.pth &nbsp;&nbsp;&nbsp;&nbsp; # Stores the weight of actor network in PPO2Agent_easy  
│   └── PPO2_model_critic_easy.pth &nbsp;&nbsp;&nbsp;&nbsp; # Stores the weight of actor network in PPO2Agent_easy  
├── runs\PPO2_training &nbsp;&nbsp;&nbsp;&nbsp; # The output file from TensorBoard(runs\PPO2_training) is a directory that contains event files, which store logged data in a format TensorBoard can read.   
│   └── events.out.tfevents.1732253638.LAPTOP-4LU1HCLC &nbsp;&nbsp;&nbsp;&nbsp; # events file of PPO2 for loss and reward diagrams, which can be read by TensorBoard  
│   └── events.out.tfevents.1732252202.sunnny &nbsp;&nbsp;&nbsp;&nbsp; # events file of PPO2_easy for loss and reward diagrams, which can be read by TensorBoard   
├── Videos &nbsp;&nbsp;&nbsp;&nbsp; # Render and simulate visual videos  
│   └── Video_PPO2.mp4 &nbsp;&nbsp;&nbsp;&nbsp; # PPO2 rendering video  
│   └── Video_A_star.mp4 &nbsp;&nbsp;&nbsp;&nbsp; # A* rendering video  
│   └── Video_PPO2_easy.mp4 &nbsp;&nbsp;&nbsp;&nbsp; # PPO2_easy rendering video  
├── environment.hml &nbsp;&nbsp;&nbsp;&nbsp; # Conda dependencies file    
├── Group07_Final_Reports &nbsp;&nbsp;&nbsp;&nbsp; # Report file  
│   └── SUNYINING_A0310458J_Report.pdf &nbsp;&nbsp;&nbsp;&nbsp; # SUN YINING Report file  
│   └── GAOYILIN_A0303608L_Report.pdf &nbsp;&nbsp;&nbsp;&nbsp; # GAO YILIN Report file  
│   └── LINHAOYU_A0304971A_Report.pdf &nbsp;&nbsp;&nbsp;&nbsp; # LIN HAOYU Report file  
└── README.md &nbsp;&nbsp;&nbsp;&nbsp; # Project documentation  

## Dependencies
Ensure the following dependencies are installed to run the code:
- `torch==1.10.2`

The above dependencies has been specified in the `environment.yml` file.

To install them inside a conda environment, follow the steps:
```
conda env create -n ME5418_Group7 -f environment.yml
conda activate ME5418_Group7
```

## Usage
**Running PPO2Agent.py**  
By default, if set Switch = 0, it is in train mode. If set Switch = 1, it is in test mode.
Please make sure that Switch = 1 before runing the code.
When running PPO2Agent.py, you can see the rendered video through pybullet window (if do not have pybullet, see the installation method below).
There are also recorded videos, which show our best testing results, can be found in the folder "Videos/Video_PPO2.mp4".
For loss diagrams, can use Tensorboard to read events file in the same folder (the method of using Tensoroard is shown below). 

**Running PPO2Agent_easy.py**  
By default, if set Switch = 0, it is in train mode. If set Switch = 1, it is in test mode.
Please make sure that Switch = 1 before runing the code.
When running PPO2Agent_easy.py, you can see the rendered video through pybullet window (if do not have pybullet, see the installation method below).
There are also recorded videos, which show our best testing results, can be found in the folder "Videos/Video_PPO2_easy.mp4".
For loss diagrams, can use Tensorboard to read events file in the same folder (the method of using Tensoroard is shown below). 

**Running a_star_agent.py**  
When running a_star_agent.py, you can see the rendered video.
There are also recorded videos, which show our best testing results, can be found in the folder "Videos/Video_A_star.mp4".


# Pybullet  
**Firstly install pybullet library in terminal.**  
`pip install pybullet`  
**You can see the graphical interface of simulation by pybullet. There is a demo video in our file.**  

# TensorboardX  
**To run our code, you need to install tensorboard first. You need to follow steps below:**
1. `conda activate <your_conda_environment>`  
2. `conda install tensorboard`
3. `conda install tensorboardX`
5. In terminal, enter `cd <your_project_directory>`
6. Enter `tensorboard --logdir runs`
7. You will get a link, open it in your browser.Then you will see the loss and reward diagrams in real time.
8. If you encouter some problems when openning the tensorboard, you may need to change the link to 'http://localhost:6006'
