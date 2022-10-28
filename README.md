## Visual_MARL

**Note:** All code and demonstrations are used for our submitted papers:
> Ruyu Luo, Wanli Ni, and Hui Tian, "Visualizing Multi-Agent Reinforcement Learning for Robotic Communication in Industrial IoT Networks," IEEE INFOCOM Demo, May. 2022.
> Ruyu Luo, Wanli Ni, Hui Tian, Julian Cheng, and Kwang-Cheng Chen, "Joint Trajectory and Radio Resource Optimization by Multi-Agent Reinforcement Learning for Autonomous Mobile Robots in Industrial Internet of Things", submitted to IEEE TVT, Nov. 2022.

In this paper, we present the simulation and visualization of multi-agent reinforcement learning (MARL).

### Representative visualization results
+ Here are four demonstrations for different stages in the MARL training process.
  + the beginning of training  <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo2.gif" alt="show" height="200" width="200" /> 
  + 800 rounds of training &emsp; <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo3.gif" alt="show" height="200" width="200" />
  + 1600 rounds of training &nbsp; <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo4.gif" alt="show" height="200" width="200" /> 
  + the end of training &emsp;&emsp;&nbsp;<img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo1.gif" alt="show" height="200" width="200" />

### Introduction to the code
+ Here is a simple introduction to the code used in our paper.
    + #### visualization tool
      + [visualization tool.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/MARL%20convergence.py):  &nbsp; Main code of four robots, connections between the environment and learning agents
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/RL_brain.py):  &nbsp; One learning agent with upper-confidence bound (UCB) exploration
      + [plot_figure.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/plot_figure.py):  &nbsp; Reward convergence figure

    + #### MARL convergence
      + [MARL convergence.py](https://github.com/lry-bupt/Visual_MARL/tree/main/visualization%20tool):  &nbsp; Main code of six robots with experience exchange, connections between the environment and learning agents & the visualization of real-time system status
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/RL_brain.py): &nbsp;  One learning agent with upper-confidence bound (UCB) exploration

    + #### robot trajectory
      + [robot_trajectory.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/robot%20trajectory.py):  &nbsp; Main code of two robots, connections between the environment and learning agents
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/RL_brain.py):  &nbsp; One learning agent with upper-confidence bound (UCB) exploration
      + [plot_figure.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/plot_figure.py):  &nbsp; The trajectories with different reward policy

### References 
[1] D. C. Nguyen et al., “[6G Internet of Things: A Comprehensive Survey](https://ieeexplore.ieee.org/abstract/document/9509294),” IEEE Internet of Things J., vol. 9, no. 1, pp. 359-383, Jan. 2022.

[2] R. Luo, H. Tian and W. Ni, “[Communication-Aware Path Design for Indoor Robots Exploiting Federated Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9569440),” in Proc. IEEE PIMRC, Helsinki, Finland, Sept. 2021, pp. 1197-1202.

[3] C. Jin et al., “[Is Q-learning Provably Efficient?](https://dl.acm.org/doi/abs/10.5555/3327345.3327395)” in Proc. NeurIPS, Montr´eal, Canada, Dec. 2018, pp. 4868-4878.
