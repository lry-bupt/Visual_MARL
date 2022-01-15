# Visual_MARL
The visualization and simulation of multi-agent reinforcement learning (MARL) with upper-confidence bound (UCB) exploration.

+ ### Representative visualization stages
  + The beginning of training <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo2.gif" alt="show" height="200" width="200" /> 
  
  + 800 rounds of training &emsp; <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo3.gif" alt="show" height="200" width="200" />

  + 1600 rounds of training &nbsp; <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo4.gif" alt="show" height="200" width="200" /> 

  + The end of training &emsp;&emsp; <img src="https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/result/demo1.gif" alt="show" height="200" width="200" />
 
+ ### A simple introduction to the code
    + #### visualization tool
      + [visualization tool.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/MARL%20convergence.py):  &nbsp; Mian code of four robots, connections between the environment and learning agents
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/RL_brain.py):  &nbsp; One learning agent with upper-confidence bound (UCB) exploration
      + [plot_figure.py](https://github.com/lry-bupt/Visual_MARL/blob/main/MARL%20convergence/plot_figure.py):  &nbsp; Reward convergence figure

    + #### MARL convergence
      + [MARL convergence.py](https://github.com/lry-bupt/Visual_MARL/tree/main/visualization%20tool):  &nbsp; Mian code of six robots with experience exchange, connections between the environment and learning agents & the visualization of real-time system status
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/visualization%20tool/RL_brain.py): &nbsp;  One learning agent with upper-confidence bound (UCB) exploration

    + #### radio map-guided path planning
      + [robot_trajectory.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/robot%20trajectory.py):  &nbsp; Mian code of two robots, connections between the environment and learning agents
      + [RL_brain.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/RL_brain.py):  &nbsp; One learning agent with upper-confidence bound (UCB) exploration
      + [plot_figure.py](https://github.com/lry-bupt/Visual_MARL/blob/main/robot%20trajectory/plot_figure.py):  &nbsp; The trajectories with different reward policy

+ ### Reference 
  + [1] [D. C. Nguyen et al., “6G Internet of Things: A Comprehensive Survey,” IEEE Internet of Things J., vol. 9, no. 1, pp. 359-383, Jan. 2022.](https://ieeexplore.ieee.org/abstract/document/9509294)
  + [2] [R. Luo, H. Tian and W. Ni, “Communication-Aware Path Design for Indoor Robots Exploiting Federated Deep Reinforcement Learning,” in Proc. IEEE PIMRC, Helsinki, Finland, Sept. 2021, pp. 1197-1202.](https://ieeexplore.ieee.org/document/9569440)
  + [3] [C. Jin et al., “Is Q-learning Provably Efficient?” in Proc. NeurIPS, Montr´eal, Canada, Dec. 2018, pp. 4868-4878.](https://dl.acm.org/doi/abs/10.5555/3327345.3327395)

+ ### Our related work
  + Ruyu Luo, Wanli Ni, and Hui Tian, "Visualizing Multi-Agent Reinforcement Learning for Robotic Communication in Industrial IoT Networks," submitted to IEEE INFOCOM Demo, Jan. 2022.
