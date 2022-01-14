# Visual_MARL
The visualization of a multi-agent reinforcement learning (MARL)-based strategy with efficient exploration strategy

+ ### Representative visualization stages

<img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo2.gif" alt="show" height="400" width="400" />  &ensp; <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo3.gif" alt="show" height="400" width="400" />

The beginning of training  &emsp; &emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp; &emsp; &emsp;&emsp;&emsp;&emsp; &emsp;  800 rounds of training

<img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo4.gif" alt="show" height="400" width="400" />  &ensp;  <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo1.gif" alt="show" height="400" width="400" />

 1600 rounds of training   &emsp; &emsp;&emsp;&emsp; &emsp; &emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp; &emsp;   The end of training  
 
 
+ ### A simple introduction to the code
    + #### visualization tool
      + [visualization tool.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/MARL%20convergence.py): mian code of four robots, connections between the environment and learning agents
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/RL_brain.py): one learning agent with upper-confidence bound (UCB) exploration
      + [plot_figure.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/plot_figure.py): reward convergence figure

    + #### MARL convergence
      + [MARL convergence.py](https://github.com/lry-bupt/Visual_MARL/tree/main/visualization%20tool): Mian code of six robots with experience exchange, connections between the environment and learning agents & the visualization of real-time system status
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/RL_brain.py): One learning agent with upper-confidence bound (UCB) exploration

    + #### robot trajectory
      + [robot_trajectory.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/robot%20trajectory.py): Mian code of two robots, connections between the environment and learning agents
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/RL_brain.py): One learning agent with upper-confidence bound (UCB) exploration
      + [plot_figure.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/plot_figure.py): The trajectories with different reward policy
