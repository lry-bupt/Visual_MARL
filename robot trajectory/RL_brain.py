"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

ACTION_SHAPE = 5*20
xy_squre = 20
h_count_done = 400

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(h_count_done*np.ones([xy_squre*xy_squre,ACTION_SHAPE]), columns=self.actions, dtype=np.float64)
        # fpr 
        # if state not in self.q_table.index:
        #     # append new state to q table
        #     self.q_table = self.q_table.append(
        #         pd.Series(
        #             [0]*len(self.actions),
        #             index=self.q_table.columns,
        #             name=state,
        #         )
        #     )
        # print(self.q_table)
        self.b_h = 0

    def choose_action(self, observation):
        # self.check_state_exist(observation)
        # action selection
        # if np.random.uniform() < self.epsilon:
        # only choose best action
        state_action = self.q_table.loc[observation, :]
        # some actions may have the same value, randomly choose on in these actions
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        # else:
        #     # choose random action
        #     action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, learning_rate, b_kh, is_done):
        self.lr = learning_rate
        self.b_h = b_kh
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        #V_khplus1 = 
        # if is_done != True:
        q_target = r + self.gamma * min( h_count_done, self.q_table.loc[s_, :].max()) + self.b_h # next state is not terminal
        # else:
        #     q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        a = 1
        # if state not in self.q_table.index:
        #     # append new state to q table
        #     self.q_table = self.q_table.append(
        #         pd.Series(
        #             [0]*len(self.actions),
        #             index=self.q_table.columns,
        #             name=state,
        #         )
        #     )