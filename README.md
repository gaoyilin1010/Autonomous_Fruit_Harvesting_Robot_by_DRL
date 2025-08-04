# Autonomous_Fruit_Harvesting_Robot_by_DRL
ME5418 Project

## Overview
This section implements a RL agent based on the PPO2 algorithm. 
The goals of this project are to train a fruit-picking robot by PPO2 algorithm to learn how to perform effective picking operations in a three-dimensional discrete space and to show a rendered video of the robot executing the policy.


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
