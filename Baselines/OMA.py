# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:01:53 2021

@author: LRY
"""
#from maze_env import Maze
from RL_brain import QLearningTable
import numpy as np
import gym
import random
import math 
from scipy.io import loadmat
import numpy.linalg as lg
import time 

env = gym.make('LargeGridWorld-v0')
env = env.unwrapped
# env2 = gym.make('LargeGridWorld-v0')
# env2 = env2.unwrapped
env3 = gym.make('LargeGridWorld-v0')
env3 = env3.unwrapped
# env4 = gym.make('LargeGridWorld-v0')
# env4 = env4.unwrapped
env5 = gym.make('LargeGridWorld-v0')
env5 = env5.unwrapped
# env6 = gym.make('LargeGridWorld-v0')
# env6 = env6.unwrapped
#env = Maze()

C_K = 3
K = 2
M_k = 3
Antenna_L = 2
ACTION_SHAPE = 5*20
P_SN_mav_array = [0.12589254117941667, 0.1584893192461113, 0.19952623149688797,0.251188643150958,0.3162277660168379,0.3981071705534972]
#ACTION_SHAPE = (5*20)**K
p_6 = np.arange(1, 7)
p_c63 = np.zeros([20,3])
p_index = 0
# P_SN_mav = 0.2
Delta_xy = 1.5

current_time = time.time()
h_count_done = 400
n_count_done = 5000
xy_squre = 20
pho_k1 = 0.01
# pho_k2 = 0.01
pho_k3 = 0.01
# pho_k4 = 0.01
pho_k5 = 0.01
# pho_k6 = 0.01

for p_1 in range(1, 5):
        for p_2 in range(p_1+1, 6):
            for p_3 in range(p_2+1, 7):
                p_c63[p_index] = [p_1, p_2, p_3]
                p_index+=1
                
for p_count in range(len(P_SN_mav_array)):
    # np.random.seed(seed_array[seed_i])
    # h_count_done = h_count_array[dr_i]
    P_SN_mav = P_SN_mav_array[p_count]
    
    RL1 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL2 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    RL3 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL4 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    RL5 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL6 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    
    N_k1_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    # N_k2_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k3_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    # N_k4_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k5_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    # N_k6_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    
    
    
    #print(RL.q_table)
    # d_k1_destination_max = (math.sqrt((xy_squre)*(xy_squre)*2))*Delta_xy
    
    for episode in range(n_count_done):
        # initial observation_k1
        ep_R_k1 = 0
        # ep_R_k2 = 0
        ep_R_k3 = 0
        # ep_R_k4 = 0
        ep_R_k5 = 0
        # ep_R_k6 = 0
        ep_reward_k1=0
        observation_k1, [x_k1_destination, y_k1_destination] = env.reset()
        # print(observation_k1, [x_k1_destination, y_k1_destination])
        x_1 = observation_k1%xy_squre
        y_1 = int(observation_k1/xy_squre)
        d_k1_destination_max = (math.sqrt((x_1-x_k1_destination)*(x_1-x_k1_destination)+(y_1-y_k1_destination)*(y_1-y_k1_destination)))*Delta_xy
        d_k1_destination_ = (math.sqrt((x_1-x_k1_destination)*(x_1-x_k1_destination)+(y_1-y_k1_destination)*(y_1-y_k1_destination)))*Delta_xy
        label_done = 0
        # ep_reward_k12=0
        # ep_reward_k2=0
        # observation_k2, [x_k2_destination, y_k2_destination] = env2.reset()
        # # print(observation_k2, [x_k2_destination, y_k2_destination])
        # x_2 = observation_k2%xy_squre
        # y_2 = int(observation_k2/xy_squre)
        # d_k2_destination_max = (math.sqrt((x_2-x_k2_destination)*(x_2-x_k2_destination)+(y_2-y_k2_destination)*(y_2-y_k2_destination)))*Delta_xy
        # d_k2_destination_ = (math.sqrt((x_2-x_k2_destination)*(x_2-x_k2_destination)+(y_2-y_k2_destination)*(y_2-y_k2_destination)))*Delta_xy
        label_done = 0
        
        k1_done_label = 0
        k2_done_label = 0
        h_count_k1_done = h_count_done
        # h_count_k2_done = h_count_done
        
        
        ep_reward_k3=0
        observation_k3, [x_k3_destination, y_k3_destination] = env3.reset()
        # print(observation_k3, [x_k3_destination, y_k3_destination])
        x_3 = observation_k3%xy_squre
        y_3 = int(observation_k3/xy_squre)
        d_k3_destination_max = (math.sqrt((x_3-x_k3_destination)*(x_3-x_k3_destination)+(y_3-y_k3_destination)*(y_3-y_k3_destination)))*Delta_xy
        d_k3_destination_ = (math.sqrt((x_3-x_k3_destination)*(x_3-x_k3_destination)+(y_3-y_k3_destination)*(y_3-y_k3_destination)))*Delta_xy
        # label_done = 0
        # ep_reward_k4=0
        # observation_k4, [x_k4_destination, y_k4_destination] = env4.reset()
        # # print(observation_k4, [x_k4_destination, y_k4_destination])
        # x_4 = observation_k4%xy_squre
        # y_4 = int(observation_k4/xy_squre)
        # d_k4_destination_max = (math.sqrt((x_4-x_k4_destination)*(x_4-x_k4_destination)+(y_4-y_k4_destination)*(y_4-y_k4_destination)))*Delta_xy
        # d_k4_destination_ = (math.sqrt((x_4-x_k4_destination)*(x_4-x_k4_destination)+(y_4-y_k4_destination)*(y_4-y_k4_destination)))*Delta_xy
        # label_done = 0
        
        k3_done_label = 0
        # k4_done_label = 0
        h_count_k3_done = h_count_done
        # h_count_k4_done = h_count_done
        
        ep_reward_k5=0
        observation_k5, [x_k5_destination, y_k5_destination] = env5.reset()
        x_5 = observation_k5%xy_squre
        y_5 = int(observation_k5/xy_squre)
        d_k5_destination_max = (math.sqrt((x_5-x_k5_destination)*(x_5-x_k5_destination)+(y_5-y_k5_destination)*(y_5-y_k5_destination)))*Delta_xy
        d_k5_destination_ = (math.sqrt((x_5-x_k5_destination)*(x_5-x_k5_destination)+(y_5-y_k5_destination)*(y_5-y_k5_destination)))*Delta_xy
        # label_done = 0
        # ep_reward_k6=0
        # observation_k6, [x_k6_destination, y_k6_destination] = env6.reset()
        # x_6 = observation_k6%xy_squre
        # y_6 = int(observation_k6/xy_squre)
        # d_k6_destination_max = (math.sqrt((x_6-x_k6_destination)*(x_6-x_k6_destination)+(y_6-y_k6_destination)*(y_6-y_k6_destination)))*Delta_xy
        # d_k6_destination_ = (math.sqrt((x_6-x_k6_destination)*(x_6-x_k6_destination)+(y_6-y_k6_destination)*(y_6-y_k6_destination)))*Delta_xy
        # label_done = 0
        
        k5_done_label = 0
        # k6_done_label = 0
        h_count_k5_done = h_count_done
        # h_count_k6_done = h_count_done
        # h_count_done = 8000
        #print('************************************************************************************')
        for h_count in range(h_count_done):
            # fresh env
            # env.render()
            # env2.render()
            
            #channel model
            x_1 = observation_k1%xy_squre
            y_1 = int(observation_k1/xy_squre)
            [x_SN1, y_SN1] = [0, 15]
            [x_SN2, y_SN2] = [5, 17]
            [x_SN3, y_SN3] = [13, 17]
            [x_SN4, y_SN4] = [3, 1]
            [x_SN5, y_SN5] = [11, 3]
            [x_SN6, y_SN6] = [15, 17]
            
            d_k1_SN1 = (math.sqrt((x_1-x_SN1)*(x_1-x_SN1)+(y_1-y_SN1)*(y_1-y_SN1)))*Delta_xy+1
            d_k1_SN2 = (math.sqrt((x_1-x_SN2)*(x_1-x_SN2)+(y_1-y_SN2)*(y_1-y_SN2)))*Delta_xy+1
            d_k1_SN3 = (math.sqrt((x_1-x_SN3)*(x_1-x_SN3)+(y_1-y_SN3)*(y_1-y_SN3)))*Delta_xy+1
            # d_k1_SN4 = (math.sqrt((x_1-x_SN4-xy_squre)*(x_1-x_SN4-xy_squre)+(y_1-y_SN4)*(y_1-y_SN4)))*Delta_xy+1
            # d_k1_SN5 = (math.sqrt((x_1-x_SN5-xy_squre)*(x_1-x_SN5-xy_squre)+(y_1-y_SN5)*(y_1-y_SN5)))*Delta_xy+1
            # d_k1_SN6 = (math.sqrt((x_1-x_SN6-xy_squre)*(x_1-x_SN6-xy_squre)+(y_1-y_SN6)*(y_1-y_SN6)))*Delta_xy+1  
            d_k1_destination = (math.sqrt((x_1-x_k1_destination)*(x_1-x_k1_destination)+(y_1-y_k1_destination)*(y_1-y_k1_destination)))*Delta_xy
            beta_k1_SN1 = math.pow(10, -3)*math.pow(d_k1_SN1, 2.2)
            beta_k1_SN2 = math.pow(10, -3)*math.pow(d_k1_SN2, 2.2)
            beta_k1_SN3 = math.pow(10, -3)*math.pow(d_k1_SN3, 2.2) 
            # beta_k1_SN4 = math.pow(10, -3)*math.pow(d_k1_SN4, 2.2)
            # beta_k1_SN5 = math.pow(10, -3)*math.pow(d_k1_SN5, 2.2)
            # beta_k1_SN6 = math.pow(10, -3)*math.pow(d_k1_SN6, 2.2) 
            theta_k1_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_1)/d_k1_SN1))
            theta_k1_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_1)/d_k1_SN2))
            theta_k1_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_1)/d_k1_SN3))
            # theta_k1_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4+xy_squre-x_1)/d_k1_SN4))
            # theta_k1_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5+xy_squre-x_1)/d_k1_SN5))
            # theta_k1_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6+xy_squre-x_1)/d_k1_SN6))
            g_k1_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN1)])
            g_k1_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN2)])
            g_k1_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN3)])
            # g_k1_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN4)])
            # g_k1_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN5)])
            # g_k1_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN6)])
            g_k1_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k1_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k1_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k1_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN1 = g_k1_SN1_LOS + g_k1_SN1_NLOS
            g_k1_SN2 = g_k1_SN2_LOS + g_k1_SN2_NLOS
            g_k1_SN3 = g_k1_SN3_LOS + g_k1_SN3_NLOS
            # g_k1_SN4 = g_k1_SN4_LOS + g_k1_SN4_NLOS
            # g_k1_SN5 = g_k1_SN5_LOS + g_k1_SN5_NLOS
            # g_k1_SN6 = g_k1_SN6_LOS + g_k1_SN6_NLOS
            h_k1_SN1 = math.sqrt(beta_k1_SN1)*g_k1_SN1
            h_k1_SN2 = math.sqrt(beta_k1_SN2)*g_k1_SN2
            h_k1_SN3 = math.sqrt(beta_k1_SN3)*g_k1_SN3
            # h_k1_SN4 = math.sqrt(beta_k1_SN4)*g_k1_SN4            
            # h_k1_SN5 = math.sqrt(beta_k1_SN5)*g_k1_SN5
            # h_k1_SN6 = math.sqrt(beta_k1_SN6)*g_k1_SN6
            #communication model
            H_k1_PART1 = np.array([h_k1_SN1[0],h_k1_SN1[1],h_k1_SN2[0],h_k1_SN2[1],h_k1_SN3[0],h_k1_SN3[1]]).T
            # H_k1_PART2 = np.array([h_k1_SN4[0],h_k1_SN4[1],h_k1_SN5[0],h_k1_SN5[1],h_k1_SN6[0],h_k1_SN6[1]]).T
    
    
            #communication model
            H_k1 = np.array([H_k1_PART1]).T
            # try:
            #     W_bar_k1 = H_k1.dot(lg.inv(H_k1.conj().T.dot(H_k1)))
            # except:
            #     W_bar_k1 = W_bar_k1
            # else:
            #     W_bar_k1 = H_k1.dot(lg.inv(H_k1.conj().T.dot(H_k1)))
            W_bar_k1 = H_k1.dot(lg.inv(H_k1.conj().T.dot(H_k1)))
            w_k1_k1 = W_bar_k1[:, 0]/(np.linalg.norm(W_bar_k1[:, 0]))
            # w_k1_k2 = W_bar_k1[:, 1]/(np.linalg.norm(W_bar_k1[:, 1]))
            # w_k1_SN3 = W_bar_k1[:, 2]/(np.linalg.norm(W_bar_k1[:, 2]))
            # w_k1_SN4 = W_bar_k1[:, 3]/(np.linalg.norm(W_bar_k1[:, 3]))
            # w_k1_SN5 = W_bar_k1[:, 4]/(np.linalg.norm(W_bar_k1[:, 4]))
            # w_k1_SN6 = W_bar_k1[:, 5]/(np.linalg.norm(W_bar_k1[:, 5]))
            # w_check1 = np.linalg.norm(w_k1_SN1)
            # w_check2 = np.linalg.norm(w_k1_SN2)
            # w_check3 = np.linalg.norm(w_k1_SN3)
            # w_k1_SN6=w_k1_k2[4:6]
            # w_k1_SN5=w_k1_k2[2:4]
            # w_k1_SN4=w_k1_k2[0:2]
            w_k1_SN3=w_k1_k1[4:6]
            w_k1_SN2=w_k1_k1[2:4]
            w_k1_SN1=w_k1_k1[0:2]
            # w_check1 = np.linalg.norm(w_k1_SN1)
            # w_check2 = np.linalg.norm(w_k1_SN2)
            # w_check3 = np.linalg.norm(w_k1_SN3)
            order_k1_SN1 = w_k1_SN1.conj().T.dot(h_k1_SN1)
            order_k1_SN2 = w_k1_SN2.conj().T.dot(h_k1_SN2)
            order_k1_SN3 = w_k1_SN3.conj().T.dot(h_k1_SN3)
            order_k1_SN1_positive = np.linalg.norm(order_k1_SN1)
            order_k1_SN2_positive = np.linalg.norm(order_k1_SN2)
            order_k1_SN3_positive = np.linalg.norm(order_k1_SN3)
            # print(order_k1_SN1_positive, order_k1_SN2_positive, order_k1_SN3_positive)
    
            # x_2 = observation_k2%xy_squre
            # y_2 = int(observation_k2/xy_squre)
            # d_k2_SN4 = (math.sqrt((x_2-x_SN4)*(x_2-x_SN4)+(y_2-y_SN4)*(y_2-y_SN4)))*Delta_xy+1
            # d_k2_SN5 = (math.sqrt((x_2-x_SN5)*(x_2-x_SN5)+(y_2-y_SN5)*(y_2-y_SN5)))*Delta_xy+1
            # d_k2_SN6 = (math.sqrt((x_2-x_SN6)*(x_2-x_SN6)+(y_2-y_SN6)*(y_2-y_SN6)))*Delta_xy+1
            # d_k2_SN1 = (math.sqrt((x_2+xy_squre-x_SN1)*(x_2+xy_squre-x_SN1)+(y_2-y_SN1)*(y_2-y_SN1)))*Delta_xy+1
            # d_k2_SN2 = (math.sqrt((x_2+xy_squre-x_SN2)*(x_2+xy_squre-x_SN2)+(y_2-y_SN2)*(y_2-y_SN2)))*Delta_xy+1
            # d_k2_SN3 = (math.sqrt((x_2+xy_squre-x_SN3)*(x_2+xy_squre-x_SN3)+(y_2-y_SN3)*(y_2-y_SN3)))*Delta_xy+1
            # d_k2_destination = (math.sqrt((x_2-x_k2_destination)*(x_2-x_k2_destination)+(y_2-y_k2_destination)*(y_2-y_k2_destination)))*Delta_xy
            # beta_k2_SN4 = math.pow(10, -3)*math.pow(d_k2_SN4, 2.2)
            # beta_k2_SN5 = math.pow(10, -3)*math.pow(d_k2_SN5, 2.2)
            # beta_k2_SN6 = math.pow(10, -3)*math.pow(d_k2_SN6, 2.2)
            # beta_k2_SN1 = math.pow(10, -3)*math.pow(d_k2_SN1, 2.2)
            # beta_k2_SN2 = math.pow(10, -3)*math.pow(d_k2_SN2, 2.2)
            # beta_k2_SN3 = math.pow(10, -3)*math.pow(d_k2_SN3, 2.2)
            # theta_k2_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_2)/d_k2_SN4))
            # theta_k2_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_2)/d_k2_SN5))
            # theta_k2_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_2)/d_k2_SN6))
            # theta_k2_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_2-xy_squre)/d_k2_SN1))
            # theta_k2_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_2-xy_squre)/d_k2_SN2))
            # theta_k2_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_2-xy_squre)/d_k2_SN3))
            # g_k2_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN4)])
            # g_k2_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN5)])
            # g_k2_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN6)])
            # g_k2_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN1)])
            # g_k2_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN2)])
            # g_k2_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN3)])
            # g_k2_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k2_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k2_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k2_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k2_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k2_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k2_SN4 = g_k2_SN4_LOS + g_k2_SN4_NLOS
            # g_k2_SN5 = g_k2_SN5_LOS + g_k2_SN5_NLOS
            # g_k2_SN6 = g_k2_SN6_LOS + g_k2_SN6_NLOS
            # g_k2_SN1 = g_k2_SN1_LOS + g_k2_SN1_NLOS
            # g_k2_SN2 = g_k2_SN2_LOS + g_k2_SN2_NLOS
            # g_k2_SN3 = g_k2_SN3_LOS + g_k2_SN3_NLOS
            # h_k2_SN4 = math.sqrt(beta_k2_SN4)*g_k2_SN4            
            # h_k2_SN5 = math.sqrt(beta_k2_SN5)*g_k2_SN5
            # h_k2_SN6 = math.sqrt(beta_k2_SN6)*g_k2_SN6
            # h_k2_SN1 = math.sqrt(beta_k2_SN1)*g_k2_SN1            
            # h_k2_SN2 = math.sqrt(beta_k2_SN2)*g_k2_SN2
            # h_k2_SN3 = math.sqrt(beta_k2_SN3)*g_k2_SN3
            # #communication model
            # H_k2_PART1 = np.array([h_k2_SN1[0],h_k2_SN1[1],h_k2_SN2[0],h_k2_SN2[1],h_k2_SN3[0],h_k2_SN3[1]]).T
            # H_k2_PART2 = np.array([h_k2_SN4[0],h_k2_SN4[1],h_k2_SN5[0],h_k2_SN5[1],h_k2_SN6[0],h_k2_SN6[1]]).T
            
            # #communication model
            # H_k2 = np.array([H_k2_PART1, H_k2_PART2]).T
            # # try:
            # #     W_bar_k2 = H_k2.dot(lg.inv(H_k2.conj().T.dot(H_k2)))
            # # except:
            # #     W_bar_k2 = W_bar_k2
            # # else:
            # #     W_bar_k2 = H_k2.dot(lg.inv(H_k2.conj().T.dot(H_k2)))
            # W_bar_k2 = H_k2.dot(lg.inv(H_k2.conj().T.dot(H_k2)))
            # w_k2_k1 = W_bar_k2[:, 0]/(np.linalg.norm(W_bar_k2[:, 0]))
            # w_k2_k2 = W_bar_k2[:, 1]/(np.linalg.norm(W_bar_k2[:, 1]))
            # # w_k2_SN6 = W_bar_k2[:, 2]/(np.linalg.norm(W_bar_k2[:, 2]))
            # # w_k2_SN1 = W_bar_k2[:, 3]/(np.linalg.norm(W_bar_k2[:, 3]))
            # # w_k2_SN2 = W_bar_k2[:, 4]/(np.linalg.norm(W_bar_k2[:, 4]))
            # # w_k2_SN3 = W_bar_k2[:, 5]/(np.linalg.norm(W_bar_k2[:, 5]))
            # # w_check1 = np.linalg.norm(w_k1_SN4)
            # # w_check2 = np.linalg.norm(w_k1_SN5)
            # # w_check3 = np.linalg.norm(w_k1_SN6)
            # w_k2_SN6=w_k2_k2[4:6]
            # w_k2_SN5=w_k2_k2[2:4]
            # w_k2_SN4=w_k2_k2[0:2]
            # w_k2_SN3=w_k2_k1[4:6]
            # w_k2_SN2=w_k2_k1[2:4]
            # w_k2_SN1=w_k2_k1[0:2]
            # # w_check1 = np.linalg.norm(w_k1_SN4)
            # # w_check2 = np.linalg.norm(w_k1_SN5)
            # # w_check3 = np.linalg.norm(w_k1_SN6)
            # order_k2_SN4 = w_k2_SN4.conj().T.dot(h_k2_SN4)
            # order_k2_SN5 = w_k2_SN5.conj().T.dot(h_k2_SN5)
            # order_k2_SN6 = w_k2_SN6.conj().T.dot(h_k2_SN6)
            # order_k2_SN4_positive = np.linalg.norm(order_k2_SN4)
            # order_k2_SN5_positive = np.linalg.norm(order_k2_SN5)
            # order_k2_SN6_positive = np.linalg.norm(order_k2_SN6)
            # # print(order_k1_SN1_positive, order_k1_SN5_positive, order_k1_SN3_positive)
    
    
            # observation_k12 = observation_k1 + observation_k2*xy_squre*xy_squre
            # RL choose action based on observation_k1
            if k1_done_label == 0:
                action_k1 = RL1.choose_action(observation_k1)
            # if k2_done_label == 0:
            #     action_k2 = RL2.choose_action(observation_k2)
            # action_k1 = action_k12%ACTION_SHAPE_single
            # action_k2 = int(action_k12/ACTION_SHAPE_single)
            order_k1 = [order_k1_SN1_positive, order_k1_SN2_positive, order_k1_SN3_positive]
            order_k1_index = [order_k1[0] for order_k1 in sorted(enumerate(order_k1),key=lambda i:i[1], reverse=True)]
            p_k1_SN_index = M_k
            for i in order_k1_index:
                exec('''p_k1_SN{}_int = p_c63[int(action_k1/5), p_k1_SN_index-1]
p_k1_SN{} = p_k1_SN{}_int*P_SN_mav/(6+1)
p_k1_SN_index = p_k1_SN_index-1'''.format(i+1, i+1, i+1))
            
            order_k1_index = [order_k1[0] for order_k1 in sorted(enumerate(order_k1),key=lambda i:i[1], reverse=True)]
            exec('''SINR_k1_SN{} = (order_k1_SN{}_positive**2*p_k1_SN{})/(10**(-9))
SINR_k1_SN{} = (order_k1_SN{}_positive**2*p_k1_SN{})/(10**(-9))
SINR_k1_SN{} = (order_k1_SN{}_positive**2*p_k1_SN{})/(10**(-9))'''.format(order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[1]+1,order_k1_index[1]+1,order_k1_index[1]+1,order_k1_index[0]+1,order_k1_index[0]+1,order_k1_index[0]+1))
    
#             order_k2 = [order_k2_SN4_positive, order_k2_SN5_positive, order_k2_SN6_positive]
#             order_k2_index = [order_k2[0] for order_k2 in sorted(enumerate(order_k2),key=lambda i:i[1], reverse=True)]
#             p_k2_SN_index = M_k
#             for i in order_k2_index:
#                 exec('''p_k2_SN{}_int = p_c63[int(action_k2/5), p_k2_SN_index-1]
# p_k2_SN{} = p_k2_SN{}_int*P_SN_mav/(6+1)
# p_k2_SN_index = p_k2_SN_index-1'''.format(i+4, i+4, i+4))
            
#             order_k2_index = [order_k2[0] for order_k2 in sorted(enumerate(order_k2),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/(10**(-9))
# SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/(10**(-9))
# SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/(10**(-9))'''.format(order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[1]+4,order_k2_index[1]+4,order_k2_index[1]+4,order_k2_index[0]+4,order_k2_index[0]+4,order_k2_index[0]+4))
            
            # print(np.linalg.norm(w_k2_SN4.conj().T.dot(h_k2_SN2)), np.linalg.norm(w_k2_SN5.conj().T.dot(h_k2_SN4)), np.linalg.norm(w_k2_SN4.conj().T.dot(h_k2_SN1)), np.linalg.norm(w_k2_SN6.conj().T.dot(h_k2_SN4)))
            # print(np.linalg.norm(w_k1_SN1.conj().T.dot(h_k1_SN2)), np.linalg.norm(w_k1_SN1.conj().T.dot(h_k1_SN4)), np.linalg.norm(w_k1_SN2.conj().T.dot(h_k1_SN5)), np.linalg.norm(w_k1_SN2.conj().T.dot(h_k1_SN4)))
    
            R_k1_SN1 = math.log((1+SINR_k1_SN1*3),2)/3
            R_k1_SN2 = math.log((1+SINR_k1_SN2*3),2)/3
            R_k1_SN3 = math.log((1+SINR_k1_SN3*3),2)/3
            # R_k2_SN4 = math.log((1+SINR_k2_SN4*3),2)/3
            # R_k2_SN5 = math.log((1+SINR_k2_SN5*3),2)/3
            # R_k2_SN6 = math.log((1+SINR_k2_SN6*3),2)/3
            
            # RL take action and get next observation_k1_k1 and reward
            observation_k1_, reward_k1, done1 = env.step(action_k1)
            # observation_k2_, reward_k2, done2 = env2.step(action_k2)
            #observation_k1_, reward, done = env.step(action)
            
            # reward_k1_reach = 0
            # if done:
            #     reward_k1_reach = 1
                
            # reward_k1 = ((0.0)*min((R_k1_SN1+R_k1_SN2+R_k1_SN3),100)*0.01+(1)*max(0, (d_k1_destination_max-d_k1_destination))/(d_k1_destination_max))*(0.2)+(1.5)*reward_k1_reach
            reward_k1 = (0.00005)*min((R_k1_SN1+R_k1_SN2+R_k1_SN3),50)
            # reward_k2 = (0.00005)*min((R_k2_SN4+R_k2_SN5+R_k2_SN6),50)
            # reward_k1 = min(0.5, min(0, (d_k1_destination_max-d_k1_destination)/(d_k1_destination_max)*(0.5))+(0.00001)*min((R_k1_SN1+R_k1_SN2+R_k1_SN3),100))
            #if h_count<5000000:
            #print ('1',((- d_k1_destination)/(d_k1_destination_max))*(1), done)
            #print('2', (0.00005)*min((R_k1_SN1+R_k1_SN2+R_k1_SN3),100))
            
            if done1 or k1_done_label==1:
                reward_k1 = 1
                # k1_done_label = 1
            # if done2 or k2_done_label==1:
            #     reward_k2 = 1
            # if done:
            #     if label_done==0:
            #         observation_k1_done = observation_k1
            #         action_k1_done = action_k1
            #         reward_k1_done = reward_k1
            #         observation_k1_done_ = observation_k1_
            #         h_count_done = h_count
            #         label_done = 1
            # if label_done==1:
            #         observation_k1 = observation_k1_done
            #         action_k1 = action_k1_done
            #         reward_k1 = reward_k1_done
            #         observation_k1_ =observation_k1_done_
            # print((d_k1_destination_max-d_k1_destination)/(d_k1_destination_max)*(2.0))   
            # print(reward_k1, label_done)
            
            # RL learn from this transition
            # reward_k12 = reward_k1+reward_k2
            
            
            
            
            x_3 = observation_k3%xy_squre
            y_3 = int(observation_k3/xy_squre)
            [x_SN1, y_SN1] = [0, 15]
            [x_SN2, y_SN2] = [5, 17]
            [x_SN3, y_SN3] = [13, 17]
            [x_SN4, y_SN4] = [3, 1]
            [x_SN5, y_SN5] = [11, 3]
            [x_SN6, y_SN6] = [15, 17]
            
            d_k3_SN1 = (math.sqrt((x_3-x_SN1)*(x_3-x_SN1)+(y_3-y_SN1)*(y_3-y_SN1)))*Delta_xy+1
            d_k3_SN2 = (math.sqrt((x_3-x_SN2)*(x_3-x_SN2)+(y_3-y_SN2)*(y_3-y_SN2)))*Delta_xy+1
            d_k3_SN3 = (math.sqrt((x_3-x_SN3)*(x_3-x_SN3)+(y_3-y_SN3)*(y_3-y_SN3)))*Delta_xy+1
            # d_k3_SN4 = (math.sqrt((x_3-x_SN4-xy_squre)*(x_3-x_SN4-xy_squre)+(y_3-y_SN4)*(y_3-y_SN4)))*Delta_xy+1
            # d_k3_SN5 = (math.sqrt((x_3-x_SN5-xy_squre)*(x_3-x_SN5-xy_squre)+(y_3-y_SN5)*(y_3-y_SN5)))*Delta_xy+1
            # d_k3_SN6 = (math.sqrt((x_3-x_SN6-xy_squre)*(x_3-x_SN6-xy_squre)+(y_3-y_SN6)*(y_3-y_SN6)))*Delta_xy+1  
            d_k3_destination = (math.sqrt((x_3-x_k3_destination)*(x_3-x_k3_destination)+(y_3-y_k3_destination)*(y_3-y_k3_destination)))*Delta_xy
            beta_k3_SN1 = math.pow(10, -3)*math.pow(d_k3_SN1, 2.2)
            beta_k3_SN2 = math.pow(10, -3)*math.pow(d_k3_SN2, 2.2)
            beta_k3_SN3 = math.pow(10, -3)*math.pow(d_k3_SN3, 2.2) 
            # beta_k3_SN4 = math.pow(10, -3)*math.pow(d_k3_SN4, 2.2)
            # beta_k3_SN5 = math.pow(10, -3)*math.pow(d_k3_SN5, 2.2)
            # beta_k3_SN6 = math.pow(10, -3)*math.pow(d_k3_SN6, 2.2) 
            theta_k3_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_3)/d_k3_SN1))
            theta_k3_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_3)/d_k3_SN2))
            theta_k3_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_3)/d_k3_SN3))
            # theta_k3_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4+xy_squre-x_3)/d_k3_SN4))
            # theta_k3_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5+xy_squre-x_3)/d_k3_SN5))
            # theta_k3_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6+xy_squre-x_3)/d_k3_SN6))
            g_k3_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN1)])
            g_k3_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN2)])
            g_k3_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN3)])
            # g_k3_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN4)])
            # g_k3_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN5)])
            # g_k3_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN6)])
            g_k3_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k3_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k3_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k3_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k3_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k3_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k3_SN1 = g_k3_SN1_LOS + g_k3_SN1_NLOS
            g_k3_SN2 = g_k3_SN2_LOS + g_k3_SN2_NLOS
            g_k3_SN3 = g_k3_SN3_LOS + g_k3_SN3_NLOS
            # g_k3_SN4 = g_k3_SN4_LOS + g_k3_SN4_NLOS
            # g_k3_SN5 = g_k3_SN5_LOS + g_k3_SN5_NLOS
            # g_k3_SN6 = g_k3_SN6_LOS + g_k3_SN6_NLOS
            h_k3_SN1 = math.sqrt(beta_k3_SN1)*g_k3_SN1
            h_k3_SN2 = math.sqrt(beta_k3_SN2)*g_k3_SN2
            h_k3_SN3 = math.sqrt(beta_k3_SN3)*g_k3_SN3
            # h_k3_SN4 = math.sqrt(beta_k3_SN4)*g_k3_SN4            
            # h_k3_SN5 = math.sqrt(beta_k3_SN5)*g_k3_SN5
            # h_k3_SN6 = math.sqrt(beta_k3_SN6)*g_k3_SN6
            #communication model
            H_k3_PART1 = np.array([h_k3_SN1[0],h_k3_SN1[1],h_k3_SN2[0],h_k3_SN2[1],h_k3_SN3[0],h_k3_SN3[1]]).T
            # H_k3_PART2 = np.array([h_k3_SN4[0],h_k3_SN4[1],h_k3_SN5[0],h_k3_SN5[1],h_k3_SN6[0],h_k3_SN6[1]]).T
    
    
            #communication model
            H_k3 = np.array([H_k3_PART1]).T
            # try:
            #     W_bar_k3 = H_k3.dot(lg.inv(H_k3.conj().T.dot(H_k3)))
            # except:
            #     W_bar_k3 = W_bar_k3
            # else:
            #     W_bar_k3 = H_k3.dot(lg.inv(H_k3.conj().T.dot(H_k3)))
            W_bar_k3 = H_k3.dot(lg.inv(H_k3.conj().T.dot(H_k3)))
            w_k3_k3 = W_bar_k3[:, 0]/(np.linalg.norm(W_bar_k3[:, 0]))
            # w_k3_k4 = W_bar_k3[:, 1]/(np.linalg.norm(W_bar_k3[:, 1]))
            # w_k3_SN3 = W_bar_k3[:, 2]/(np.linalg.norm(W_bar_k3[:, 2]))
            # w_k3_SN4 = W_bar_k3[:, 3]/(np.linalg.norm(W_bar_k3[:, 3]))
            # w_k3_SN5 = W_bar_k3[:, 4]/(np.linalg.norm(W_bar_k3[:, 4]))
            # w_k3_SN6 = W_bar_k3[:, 5]/(np.linalg.norm(W_bar_k3[:, 5]))
            # w_check3 = np.linalg.norm(w_k3_SN1)
            # w_check4 = np.linalg.norm(w_k3_SN2)
            # w_check3 = np.linalg.norm(w_k3_SN3)
            # w_k3_SN6=w_k3_k4[4:6]
            # w_k3_SN5=w_k3_k4[2:4]
            # w_k3_SN4=w_k3_k4[0:2]
            w_k3_SN3=w_k3_k3[4:6]
            w_k3_SN2=w_k3_k3[2:4]
            w_k3_SN1=w_k3_k3[0:2]
            # w_check3 = np.linalg.norm(w_k3_SN1)
            # w_check4 = np.linalg.norm(w_k3_SN2)
            # w_check3 = np.linalg.norm(w_k3_SN3)
            order_k3_SN1 = w_k3_SN1.conj().T.dot(h_k3_SN1)
            order_k3_SN2 = w_k3_SN2.conj().T.dot(h_k3_SN2)
            order_k3_SN3 = w_k3_SN3.conj().T.dot(h_k3_SN3)
            order_k3_SN1_positive = np.linalg.norm(order_k3_SN1)
            order_k3_SN2_positive = np.linalg.norm(order_k3_SN2)
            order_k3_SN3_positive = np.linalg.norm(order_k3_SN3)
            # print(order_k3_SN1_positive, order_k3_SN2_positive, order_k3_SN3_positive)
    
            # x_4 = observation_k4%xy_squre
            # y_4 = int(observation_k4/xy_squre)
            # d_k4_SN4 = (math.sqrt((x_4-x_SN4)*(x_4-x_SN4)+(y_4-y_SN4)*(y_4-y_SN4)))*Delta_xy+1
            # d_k4_SN5 = (math.sqrt((x_4-x_SN5)*(x_4-x_SN5)+(y_4-y_SN5)*(y_4-y_SN5)))*Delta_xy+1
            # d_k4_SN6 = (math.sqrt((x_4-x_SN6)*(x_4-x_SN6)+(y_4-y_SN6)*(y_4-y_SN6)))*Delta_xy+1
            # d_k4_SN1 = (math.sqrt((x_4+xy_squre-x_SN1)*(x_4+xy_squre-x_SN1)+(y_4-y_SN1)*(y_4-y_SN1)))*Delta_xy+1
            # d_k4_SN2 = (math.sqrt((x_4+xy_squre-x_SN2)*(x_4+xy_squre-x_SN2)+(y_4-y_SN2)*(y_4-y_SN2)))*Delta_xy+1
            # d_k4_SN3 = (math.sqrt((x_4+xy_squre-x_SN3)*(x_4+xy_squre-x_SN3)+(y_4-y_SN3)*(y_4-y_SN3)))*Delta_xy+1
            # d_k4_destination = (math.sqrt((x_4-x_k4_destination)*(x_4-x_k4_destination)+(y_4-y_k4_destination)*(y_4-y_k4_destination)))*Delta_xy
            # beta_k4_SN4 = math.pow(10, -3)*math.pow(d_k4_SN4, 2.2)
            # beta_k4_SN5 = math.pow(10, -3)*math.pow(d_k4_SN5, 2.2)
            # beta_k4_SN6 = math.pow(10, -3)*math.pow(d_k4_SN6, 2.2)
            # beta_k4_SN1 = math.pow(10, -3)*math.pow(d_k4_SN1, 2.2)
            # beta_k4_SN2 = math.pow(10, -3)*math.pow(d_k4_SN2, 2.2)
            # beta_k4_SN3 = math.pow(10, -3)*math.pow(d_k4_SN3, 2.2)
            # theta_k4_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_4)/d_k4_SN4))
            # theta_k4_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_4)/d_k4_SN5))
            # theta_k4_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_4)/d_k4_SN6))
            # theta_k4_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_4-xy_squre)/d_k4_SN1))
            # theta_k4_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_4-xy_squre)/d_k4_SN2))
            # theta_k4_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_4-xy_squre)/d_k4_SN3))
            # g_k4_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN4)])
            # g_k4_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN5)])
            # g_k4_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN6)])
            # g_k4_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN1)])
            # g_k4_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN2)])
            # g_k4_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN3)])
            # g_k4_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k4_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k4_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k4_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k4_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k4_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k4_SN4 = g_k4_SN4_LOS + g_k4_SN4_NLOS
            # g_k4_SN5 = g_k4_SN5_LOS + g_k4_SN5_NLOS
            # g_k4_SN6 = g_k4_SN6_LOS + g_k4_SN6_NLOS
            # g_k4_SN1 = g_k4_SN1_LOS + g_k4_SN1_NLOS
            # g_k4_SN2 = g_k4_SN2_LOS + g_k4_SN2_NLOS
            # g_k4_SN3 = g_k4_SN3_LOS + g_k4_SN3_NLOS
            # h_k4_SN4 = math.sqrt(beta_k4_SN4)*g_k4_SN4            
            # h_k4_SN5 = math.sqrt(beta_k4_SN5)*g_k4_SN5
            # h_k4_SN6 = math.sqrt(beta_k4_SN6)*g_k4_SN6
            # h_k4_SN1 = math.sqrt(beta_k4_SN1)*g_k4_SN1            
            # h_k4_SN2 = math.sqrt(beta_k4_SN2)*g_k4_SN2
            # h_k4_SN3 = math.sqrt(beta_k4_SN3)*g_k4_SN3
            # #communication model
            # H_k4_PART1 = np.array([h_k4_SN1[0],h_k4_SN1[1],h_k4_SN2[0],h_k4_SN2[1],h_k4_SN3[0],h_k4_SN3[1]]).T
            # H_k4_PART2 = np.array([h_k4_SN4[0],h_k4_SN4[1],h_k4_SN5[0],h_k4_SN5[1],h_k4_SN6[0],h_k4_SN6[1]]).T
            
            # #communication model
            # H_k4 = np.array([H_k4_PART1, H_k4_PART2]).T
            # # try:
            # #     W_bar_k4 = H_k4.dot(lg.inv(H_k4.conj().T.dot(H_k4)))
            # # except:
            # #     W_bar_k4 = W_bar_k4
            # # else:
            # #     W_bar_k4 = H_k4.dot(lg.inv(H_k4.conj().T.dot(H_k4)))
            # W_bar_k4 = H_k4.dot(lg.inv(H_k4.conj().T.dot(H_k4)))
            # w_k4_k3 = W_bar_k4[:, 0]/(np.linalg.norm(W_bar_k4[:, 0]))
            # w_k4_k4 = W_bar_k4[:, 1]/(np.linalg.norm(W_bar_k4[:, 1]))
            # # w_k4_SN6 = W_bar_k4[:, 2]/(np.linalg.norm(W_bar_k4[:, 2]))
            # # w_k4_SN1 = W_bar_k4[:, 3]/(np.linalg.norm(W_bar_k4[:, 3]))
            # # w_k4_SN2 = W_bar_k4[:, 4]/(np.linalg.norm(W_bar_k4[:, 4]))
            # # w_k4_SN3 = W_bar_k4[:, 5]/(np.linalg.norm(W_bar_k4[:, 5]))
            # # w_check3 = np.linalg.norm(w_k3_SN4)
            # # w_check4 = np.linalg.norm(w_k3_SN5)
            # # w_check3 = np.linalg.norm(w_k3_SN6)
            # w_k4_SN6=w_k4_k4[4:6]
            # w_k4_SN5=w_k4_k4[2:4]
            # w_k4_SN4=w_k4_k4[0:2]
            # w_k4_SN3=w_k4_k3[4:6]
            # w_k4_SN2=w_k4_k3[2:4]
            # w_k4_SN1=w_k4_k3[0:2]
            # # w_check3 = np.linalg.norm(w_k3_SN4)
            # # w_check4 = np.linalg.norm(w_k3_SN5)
            # # w_check3 = np.linalg.norm(w_k3_SN6)
            # order_k4_SN4 = w_k4_SN4.conj().T.dot(h_k4_SN4)
            # order_k4_SN5 = w_k4_SN5.conj().T.dot(h_k4_SN5)
            # order_k4_SN6 = w_k4_SN6.conj().T.dot(h_k4_SN6)
            # order_k4_SN4_positive = np.linalg.norm(order_k4_SN4)
            # order_k4_SN5_positive = np.linalg.norm(order_k4_SN5)
            # order_k4_SN6_positive = np.linalg.norm(order_k4_SN6)
            # # print(order_k3_SN1_positive, order_k3_SN5_positive, order_k3_SN3_positive)
    
    
            # observation_k32 = observation_k3 + observation_k4*xy_squre*xy_squre
            # RL choose action based on observation_k3
            if k3_done_label == 0:
                action_k3 = RL3.choose_action(observation_k3)
            # if k4_done_label == 0:
            #     action_k4 = RL4.choose_action(observation_k4)
            # action_k3 = action_k32%ACTION_SHAPE_single
            # action_k4 = int(action_k32/ACTION_SHAPE_single)
            order_k3 = [order_k3_SN1_positive, order_k3_SN2_positive, order_k3_SN3_positive]
            order_k3_index = [order_k3[0] for order_k3 in sorted(enumerate(order_k3),key=lambda i:i[1], reverse=True)]
            p_k3_SN_index = M_k
            for i in order_k3_index:
                exec('''p_k3_SN{}_int = p_c63[int(action_k3/5), p_k3_SN_index-1]
p_k3_SN{} = p_k3_SN{}_int*P_SN_mav/(6+1)
p_k3_SN_index = p_k3_SN_index-1'''.format(i+1, i+1, i+1))
            
            order_k3_index = [order_k3[0] for order_k3 in sorted(enumerate(order_k3),key=lambda i:i[1], reverse=True)]
            exec('''SINR_k3_SN{} = (order_k3_SN{}_positive**2*p_k3_SN{})/(10**(-9))
SINR_k3_SN{} = (order_k3_SN{}_positive**2*p_k3_SN{})/(10**(-9))
SINR_k3_SN{} = (order_k3_SN{}_positive**2*p_k3_SN{})/(10**(-9))'''.format(order_k3_index[2]+1,order_k3_index[2]+1,order_k3_index[2]+1,order_k3_index[1]+1,order_k3_index[1]+1,order_k3_index[1]+1,order_k3_index[0]+1,order_k3_index[0]+1,order_k3_index[0]+1))
    
#             order_k4 = [order_k4_SN4_positive, order_k4_SN5_positive, order_k4_SN6_positive]
#             order_k4_index = [order_k4[0] for order_k4 in sorted(enumerate(order_k4),key=lambda i:i[1], reverse=True)]
#             p_k4_SN_index = M_k
#             for i in order_k4_index:
#                 exec('''p_k4_SN{}_int = p_c63[int(action_k4/5), p_k4_SN_index-1]
# p_k4_SN{} = p_k4_SN{}_int*P_SN_mav/(6+1)
# p_k4_SN_index = p_k4_SN_index-1'''.format(i+4, i+4, i+4))
            
#             order_k4_index = [order_k4[0] for order_k4 in sorted(enumerate(order_k4),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k4_SN{} = (order_k4_SN{}_positive**2*p_k4_SN{})/(10**(-9))
# SINR_k4_SN{} = (order_k4_SN{}_positive**2*p_k4_SN{})/(10**(-9))
# SINR_k4_SN{} = (order_k4_SN{}_positive**2*p_k4_SN{})/(10**(-9))'''.format(order_k4_index[2]+4,order_k4_index[2]+4,order_k4_index[2]+4,order_k4_index[1]+4,order_k4_index[1]+4,order_k4_index[1]+4,order_k4_index[0]+4,order_k4_index[0]+4,order_k4_index[0]+4))
               
            # print(np.linalg.norm(w_k4_SN4.conj().T.dot(h_k4_SN2)), np.linalg.norm(w_k4_SN5.conj().T.dot(h_k4_SN4)), np.linalg.norm(w_k4_SN4.conj().T.dot(h_k4_SN1)), np.linalg.norm(w_k4_SN6.conj().T.dot(h_k4_SN4)))
            # print(np.linalg.norm(w_k3_SN1.conj().T.dot(h_k3_SN2)), np.linalg.norm(w_k3_SN1.conj().T.dot(h_k3_SN4)), np.linalg.norm(w_k3_SN2.conj().T.dot(h_k3_SN5)), np.linalg.norm(w_k3_SN2.conj().T.dot(h_k3_SN4)))
    
            R_k3_SN1 = math.log((1+SINR_k3_SN1*3),2)/3
            R_k3_SN2 = math.log((1+SINR_k3_SN2*3),2)/3
            R_k3_SN3 = math.log((1+SINR_k3_SN3*3),2)/3
            # R_k4_SN4 = math.log((1+SINR_k4_SN4*3),2)/3
            # R_k4_SN5 = math.log((1+SINR_k4_SN5*3),2)/3
            # R_k4_SN6 = math.log((1+SINR_k4_SN6*3),2)/3
    
            # RL take action and get next observation_k3_k3 and reward
            observation_k3_, reward_k3, done3 = env3.step(action_k3)
            # observation_k4_, reward_k4, done4 = env4.step(action_k4)
            #observation_k3_, reward, done = env.step(action)
            
            # reward_k3_reach = 0
            # if done:
            #     reward_k3_reach = 1

            # reward_k3 = ((0.0)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),100)*0.01+(1)*max(0, (d_k3_destination_max-d_k3_destination))/(d_k3_destination_max))*(0.2)+(1.5)*reward_k3_reach
            reward_k3 = (0.00005)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),50)
            # reward_k4 = (0.00005)*min((R_k4_SN4+R_k4_SN5+R_k4_SN6),50)
            # reward_k3 = min(0.5, min(0, (d_k3_destination_max-d_k3_destination)/(d_k3_destination_max)*(0.5))+(0.00001)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),100))
            #if h_count<5000000:
            #print ('1',((- d_k3_destination)/(d_k3_destination_max))*(1), done)
            #print('2', (0.00005)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),100))
            
            if done3 or k3_done_label==1:
                reward_k3 = 1
                # k3_done_label = 1
            # if done4 or k4_done_label==1:
            #     reward_k4 = 1
            # print((0.00005)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),50),reward_k2,reward_k3, reward_k4)
            # if done:
            #     if label_done==0:
            #         observation_k3_done = observation_k1
            #         action_k1_done = action_k1
            #         reward_k1_done = reward_k1
            #         observation_k1_done_ = observation_k1_
            #         h_count_done = h_count
            #         label_done = 1
            # if label_done==1:
            #         observation_k1 = observation_k1_done
            #         action_k1 = action_k1_done
            #         reward_k1 = reward_k1_done
            #         observation_k1_ =observation_k1_done_
            # print((d_k1_destination_max-d_k1_destination)/(d_k1_destination_max)*(2.0))   
            # print(reward_k1, label_done)
            
            # RL learn from this transition
            x_5 = observation_k5%xy_squre
            y_5 = int(observation_k5/xy_squre)
            [x_SN1, y_SN1] = [0, 15]
            [x_SN2, y_SN2] = [5, 17]
            [x_SN3, y_SN3] = [13, 17]
            [x_SN4, y_SN4] = [3, 1]
            [x_SN5, y_SN5] = [11, 3]
            [x_SN6, y_SN6] = [15, 17]
            
            d_k5_SN1 = (math.sqrt((x_5-x_SN1)*(x_5-x_SN1)+(y_5-y_SN1)*(y_5-y_SN1)))*Delta_xy+1
            d_k5_SN2 = (math.sqrt((x_5-x_SN2)*(x_5-x_SN2)+(y_5-y_SN2)*(y_5-y_SN2)))*Delta_xy+1
            d_k5_SN3 = (math.sqrt((x_5-x_SN3)*(x_5-x_SN3)+(y_5-y_SN3)*(y_5-y_SN3)))*Delta_xy+1
            # d_k5_SN4 = (math.sqrt((x_5-x_SN4-xy_squre)*(x_5-x_SN4-xy_squre)+(y_5-y_SN4)*(y_5-y_SN4)))*Delta_xy+1
            # d_k5_SN5 = (math.sqrt((x_5-x_SN5-xy_squre)*(x_5-x_SN5-xy_squre)+(y_5-y_SN5)*(y_5-y_SN5)))*Delta_xy+1
            # d_k5_SN6 = (math.sqrt((x_5-x_SN6-xy_squre)*(x_5-x_SN6-xy_squre)+(y_5-y_SN6)*(y_5-y_SN6)))*Delta_xy+1  
            d_k5_destination = (math.sqrt((x_5-x_k5_destination)*(x_5-x_k5_destination)+(y_5-y_k5_destination)*(y_5-y_k5_destination)))*Delta_xy
            beta_k5_SN1 = math.pow(10, -3)*math.pow(d_k5_SN1, 2.2)
            beta_k5_SN2 = math.pow(10, -3)*math.pow(d_k5_SN2, 2.2)
            beta_k5_SN3 = math.pow(10, -3)*math.pow(d_k5_SN3, 2.2) 
            # beta_k5_SN4 = math.pow(10, -3)*math.pow(d_k5_SN4, 2.2)
            # beta_k5_SN5 = math.pow(10, -3)*math.pow(d_k5_SN5, 2.2)
            # beta_k5_SN6 = math.pow(10, -3)*math.pow(d_k5_SN6, 2.2) 
            theta_k5_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_5)/d_k5_SN1))
            theta_k5_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_5)/d_k5_SN2))
            theta_k5_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_5)/d_k5_SN3))
            # theta_k5_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4+xy_squre-x_5)/d_k5_SN4))
            # theta_k5_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5+xy_squre-x_5)/d_k5_SN5))
            # theta_k5_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6+xy_squre-x_5)/d_k5_SN6))
            g_k5_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN1)])
            g_k5_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN2)])
            g_k5_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN3)])
            # g_k5_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN4)])
            # g_k5_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN5)])
            # g_k5_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN6)])
            g_k5_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k5_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k5_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k5_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k5_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k5_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k5_SN1 = g_k5_SN1_LOS + g_k5_SN1_NLOS
            g_k5_SN2 = g_k5_SN2_LOS + g_k5_SN2_NLOS
            g_k5_SN3 = g_k5_SN3_LOS + g_k5_SN3_NLOS
            # g_k5_SN4 = g_k5_SN4_LOS + g_k5_SN4_NLOS
            # g_k5_SN5 = g_k5_SN5_LOS + g_k5_SN5_NLOS
            # g_k5_SN6 = g_k5_SN6_LOS + g_k5_SN6_NLOS
            h_k5_SN1 = math.sqrt(beta_k5_SN1)*g_k5_SN1
            h_k5_SN2 = math.sqrt(beta_k5_SN2)*g_k5_SN2
            h_k5_SN3 = math.sqrt(beta_k5_SN3)*g_k5_SN3
            # h_k5_SN4 = math.sqrt(beta_k5_SN4)*g_k5_SN4            
            # h_k5_SN5 = math.sqrt(beta_k5_SN5)*g_k5_SN5
            # h_k5_SN6 = math.sqrt(beta_k5_SN6)*g_k5_SN6
            #communication model
            H_k5_PART1 = np.array([h_k5_SN1[0],h_k5_SN1[1],h_k5_SN2[0],h_k5_SN2[1],h_k5_SN3[0],h_k5_SN3[1]]).T
            # H_k5_PART2 = np.array([h_k5_SN4[0],h_k5_SN4[1],h_k5_SN5[0],h_k5_SN5[1],h_k5_SN6[0],h_k5_SN6[1]]).T
    
    
            #communication model
            H_k5 = np.array([H_k5_PART1]).T
            W_bar_k5 = H_k5.dot(lg.inv(H_k5.conj().T.dot(H_k5)))
            w_k5_k5 = W_bar_k5[:, 0]/(np.linalg.norm(W_bar_k5[:, 0]))
            # w_k5_k6 = W_bar_k5[:, 1]/(np.linalg.norm(W_bar_k5[:, 1]))
            # w_k5_SN6=w_k5_k6[4:6]
            # w_k5_SN5=w_k5_k6[2:4]
            # w_k5_SN4=w_k5_k6[0:2]
            w_k5_SN3=w_k5_k5[4:6]
            w_k5_SN2=w_k5_k5[2:4]
            w_k5_SN1=w_k5_k5[0:2]
            # w_check3 = np.linalg.norm(w_k3_SN1)
            # w_check4 = np.linalg.norm(w_k3_SN2)
            # w_check3 = np.linalg.norm(w_k3_SN3)
            order_k5_SN1 = w_k5_SN1.conj().T.dot(h_k5_SN1)
            order_k5_SN2 = w_k5_SN2.conj().T.dot(h_k5_SN2)
            order_k5_SN3 = w_k5_SN3.conj().T.dot(h_k5_SN3)
            order_k5_SN1_positive = np.linalg.norm(order_k5_SN1)
            order_k5_SN2_positive = np.linalg.norm(order_k5_SN2)
            order_k5_SN3_positive = np.linalg.norm(order_k5_SN3)
            # print(order_k3_SN1_positive, order_k3_SN2_positive, order_k3_SN3_positive)
    
            # x_6 = observation_k6%xy_squre
            # y_6 = int(observation_k6/xy_squre)
            # d_k6_SN4 = (math.sqrt((x_6-x_SN4)*(x_6-x_SN4)+(y_6-y_SN4)*(y_6-y_SN4)))*Delta_xy+1
            # d_k6_SN5 = (math.sqrt((x_6-x_SN5)*(x_6-x_SN5)+(y_6-y_SN5)*(y_6-y_SN5)))*Delta_xy+1
            # d_k6_SN6 = (math.sqrt((x_6-x_SN6)*(x_6-x_SN6)+(y_6-y_SN6)*(y_6-y_SN6)))*Delta_xy+1
            # d_k6_SN1 = (math.sqrt((x_6+xy_squre-x_SN1)*(x_6+xy_squre-x_SN1)+(y_6-y_SN1)*(y_6-y_SN1)))*Delta_xy+1
            # d_k6_SN2 = (math.sqrt((x_6+xy_squre-x_SN2)*(x_6+xy_squre-x_SN2)+(y_6-y_SN2)*(y_6-y_SN2)))*Delta_xy+1
            # d_k6_SN3 = (math.sqrt((x_6+xy_squre-x_SN3)*(x_6+xy_squre-x_SN3)+(y_6-y_SN3)*(y_6-y_SN3)))*Delta_xy+1
            # d_k6_destination = (math.sqrt((x_6-x_k6_destination)*(x_6-x_k6_destination)+(y_6-y_k6_destination)*(y_6-y_k6_destination)))*Delta_xy
            # beta_k6_SN4 = math.pow(10, -3)*math.pow(d_k6_SN4, 2.2)
            # beta_k6_SN5 = math.pow(10, -3)*math.pow(d_k6_SN5, 2.2)
            # beta_k6_SN6 = math.pow(10, -3)*math.pow(d_k6_SN6, 2.2)
            # beta_k6_SN1 = math.pow(10, -3)*math.pow(d_k6_SN1, 2.2)
            # beta_k6_SN2 = math.pow(10, -3)*math.pow(d_k6_SN2, 2.2)
            # beta_k6_SN3 = math.pow(10, -3)*math.pow(d_k6_SN3, 2.2)
            # theta_k6_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_6)/d_k6_SN4))
            # theta_k6_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_6)/d_k6_SN5))
            # theta_k6_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_6)/d_k6_SN6))
            # theta_k6_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_6-xy_squre)/d_k6_SN1))
            # theta_k6_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_6-xy_squre)/d_k6_SN2))
            # theta_k6_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_6-xy_squre)/d_k6_SN3))
            # g_k6_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN4)])
            # g_k6_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN5)])
            # g_k6_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN6)])
            # g_k6_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN1)])
            # g_k6_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN2)])
            # g_k6_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN3)])
            # g_k6_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k6_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k6_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k6_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k6_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k6_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            # g_k6_SN4 = g_k6_SN4_LOS + g_k6_SN4_NLOS
            # g_k6_SN5 = g_k6_SN5_LOS + g_k6_SN5_NLOS
            # g_k6_SN6 = g_k6_SN6_LOS + g_k6_SN6_NLOS
            # g_k6_SN1 = g_k6_SN1_LOS + g_k6_SN1_NLOS
            # g_k6_SN2 = g_k6_SN2_LOS + g_k6_SN2_NLOS
            # g_k6_SN3 = g_k6_SN3_LOS + g_k6_SN3_NLOS
            # h_k6_SN4 = math.sqrt(beta_k6_SN4)*g_k6_SN4            
            # h_k6_SN5 = math.sqrt(beta_k6_SN5)*g_k6_SN5
            # h_k6_SN6 = math.sqrt(beta_k6_SN6)*g_k6_SN6
            # h_k6_SN1 = math.sqrt(beta_k6_SN1)*g_k6_SN1            
            # h_k6_SN2 = math.sqrt(beta_k6_SN2)*g_k6_SN2
            # h_k6_SN3 = math.sqrt(beta_k6_SN3)*g_k6_SN3
            # #communication model
            # H_k6_PART1 = np.array([h_k6_SN1[0],h_k6_SN1[1],h_k6_SN2[0],h_k6_SN2[1],h_k6_SN3[0],h_k6_SN3[1]]).T
            # H_k6_PART2 = np.array([h_k6_SN4[0],h_k6_SN4[1],h_k6_SN5[0],h_k6_SN5[1],h_k6_SN6[0],h_k6_SN6[1]]).T
            
            # #communication model
            # H_k6 = np.array([H_k6_PART1, H_k6_PART2]).T
            # W_bar_k6 = H_k6.dot(lg.inv(H_k6.conj().T.dot(H_k6)))
            # w_k6_k5 = W_bar_k6[:, 0]/(np.linalg.norm(W_bar_k6[:, 0]))
            # w_k6_k6 = W_bar_k6[:, 1]/(np.linalg.norm(W_bar_k6[:, 1]))
            # w_k6_SN6=w_k6_k6[4:6]
            # w_k6_SN5=w_k6_k6[2:4]
            # w_k6_SN4=w_k6_k6[0:2]
            # w_k6_SN3=w_k6_k5[4:6]
            # w_k6_SN2=w_k6_k5[2:4]
            # w_k6_SN1=w_k6_k5[0:2]
            # # w_check3 = np.linalg.norm(w_k3_SN4)
            # # w_check4 = np.linalg.norm(w_k3_SN5)
            # # w_check3 = np.linalg.norm(w_k3_SN6)
            # order_k6_SN4 = w_k6_SN4.conj().T.dot(h_k6_SN4)
            # order_k6_SN5 = w_k6_SN5.conj().T.dot(h_k6_SN5)
            # order_k6_SN6 = w_k6_SN6.conj().T.dot(h_k6_SN6)
            # order_k6_SN4_positive = np.linalg.norm(order_k6_SN4)
            # order_k6_SN5_positive = np.linalg.norm(order_k6_SN5)
            # order_k6_SN6_positive = np.linalg.norm(order_k6_SN6)
            # print(order_k3_SN1_positive, order_k3_SN5_positive, order_k3_SN3_positive)
    
    
            # observation_k32 = observation_k3 + observation_k4*xy_squre*xy_squre
            # RL choose action based on observation_k3
            if k5_done_label == 0:
                action_k5 = RL5.choose_action(observation_k5)
            # if k6_done_label == 0:
            #     action_k6 = RL6.choose_action(observation_k6)
            # action_k3 = action_k32%ACTION_SHAPE_single
            # action_k4 = int(action_k32/ACTION_SHAPE_single)
            order_k5 = [order_k5_SN1_positive, order_k5_SN2_positive, order_k5_SN3_positive]
            order_k5_index = [order_k5[0] for order_k5 in sorted(enumerate(order_k5),key=lambda i:i[1], reverse=True)]
            p_k5_SN_index = M_k
            for i in order_k5_index:
                exec('''p_k5_SN{}_int = p_c63[int(action_k5/5), p_k5_SN_index-1]
p_k5_SN{} = p_k5_SN{}_int*P_SN_mav/(6+1)
p_k5_SN_index = p_k5_SN_index-1'''.format(i+1, i+1, i+1))
            
            order_k5_index = [order_k5[0] for order_k5 in sorted(enumerate(order_k5),key=lambda i:i[1], reverse=True)]
            exec('''SINR_k5_SN{} = (order_k5_SN{}_positive**2*p_k5_SN{})/(10**(-9))
SINR_k5_SN{} = (order_k5_SN{}_positive**2*p_k5_SN{})/(10**(-9))
SINR_k5_SN{} = (order_k5_SN{}_positive**2*p_k5_SN{})/(10**(-9))'''.format(order_k5_index[2]+1,order_k5_index[2]+1,order_k5_index[2]+1,order_k5_index[1]+1,order_k5_index[1]+1,order_k5_index[1]+1,order_k5_index[0]+1,order_k5_index[0]+1,order_k5_index[0]+1))
    
#             order_k6 = [order_k6_SN4_positive, order_k6_SN5_positive, order_k6_SN6_positive]
#             order_k6_index = [order_k6[0] for order_k6 in sorted(enumerate(order_k6),key=lambda i:i[1], reverse=True)]
#             p_k6_SN_index = M_k
#             for i in order_k6_index:
#                 exec('''p_k6_SN{}_int = p_c63[int(action_k6/5), p_k6_SN_index-1]
# p_k6_SN{} = p_k6_SN{}_int*P_SN_mav/(6+1)
# p_k6_SN_index = p_k6_SN_index-1'''.format(i+4, i+4, i+4))
            
#             order_k6_index = [order_k6[0] for order_k6 in sorted(enumerate(order_k6),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k6_SN{} = (order_k6_SN{}_positive**2*p_k6_SN{})/(10**(-9))
# SINR_k6_SN{} = (order_k6_SN{}_positive**2*p_k6_SN{})/(10**(-9))
# SINR_k6_SN{} = (order_k6_SN{}_positive**2*p_k6_SN{})/(10**(-9))'''.format(order_k6_index[2]+4,order_k6_index[2]+4,order_k6_index[2]+4,order_k6_index[1]+4,order_k6_index[1]+4,order_k6_index[1]+4,order_k6_index[0]+4,order_k6_index[0]+4,order_k6_index[0]+4))
            
            R_k5_SN1 = math.log((1+SINR_k5_SN1*3),2)/3
            R_k5_SN2 = math.log((1+SINR_k5_SN2*3),2)/3
            R_k5_SN3 = math.log((1+SINR_k5_SN3*3),2)/3
            # R_k6_SN4 = math.log((1+SINR_k6_SN4*3),2)/3
            # R_k6_SN5 = math.log((1+SINR_k6_SN5*3),2)/3
            # R_k6_SN6 = math.log((1+SINR_k6_SN6*3),2)/3
            ep_R_k1 = R_k1_SN1+R_k1_SN2+R_k1_SN3+ep_R_k1
            # ep_R_k2 = R_k2_SN4+R_k2_SN5+R_k2_SN6+ep_R_k2
            ep_R_k3 = R_k3_SN1+R_k3_SN2+R_k3_SN3+ep_R_k3
            # ep_R_k4 = R_k4_SN4+R_k4_SN5+R_k4_SN6+ep_R_k4
            ep_R_k5 = R_k5_SN1+R_k5_SN2+R_k5_SN3+ep_R_k5
            # ep_R_k6 = R_k6_SN4+R_k6_SN5+R_k6_SN6+ep_R_k6
            # RL take action and get next observation_k5_k5 and reward
            observation_k5_, reward_k5, done5 = env5.step(action_k5)
            # observation_k6_, reward_k6, done6 = env6.step(action_k6)
            #observation_k5_, reward, done = env.step(action)
            
            # reward_k5_reach = 0
            # if done:
            #     reward_k5_reach = 1
                
            # reward_k5 = ((0.0)*min((R_k5_SN1+R_k5_SN2+R_k5_SN3),100)*0.01+(1)*max(0, (d_k5_destination_max-d_k5_destination))/(d_k5_destination_max))*(0.2)+(1.5)*reward_k5_reach
            reward_k5 = (0.00005)*min((R_k5_SN1+R_k5_SN2+R_k5_SN3),50)
            # reward_k6 = (0.00005)*min((R_k6_SN4+R_k6_SN5+R_k6_SN6),50)
            
            if done5 or k5_done_label==1:
                reward_k5 = 1
                # k5_done_label = 1
            # if done6 or k6_done_label==1:
            #     reward_k6 = 1
                
            # reward_k1234 = reward_k3+reward_k4+reward_k1+reward_k2
            if k1_done_label == 0:
             # observation_k12_= observation_k1_ + observation_k2_*xy_squre*xy_squre
                 N_k1_hnsa[observation_k1, action_k1, h_count] = N_k1_hnsa[observation_k1, action_k1, h_count]+1
                 lr_k1 = (h_count_done+1)/(h_count_done+N_k1_hnsa[observation_k1, action_k1, h_count])
                 b_k1_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k1)/(N_k1_hnsa[observation_k1, action_k1, h_count]*C_K))
                 # print(reward_k1)
                 # RL learn from this transition
                 # print(lr_k1, b_k1_done)
                 RL1.learn(observation_k1, action_k1, reward_k1, observation_k1_, lr_k1, b_k1_done, done1)
                 ep_reward_k1 += reward_k1
                 observation_k1 = observation_k1_
                 h_count_k1_done = h_count
                
                 N_k3_hnsa[observation_k1, action_k1, h_count] = N_k3_hnsa[observation_k1, action_k1, h_count]+1
                 lr_k3 = (h_count_done+1)/(h_count_done+N_k3_hnsa[observation_k1, action_k1, h_count])
                 b_k3_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k3)/(N_k3_hnsa[observation_k1, action_k1, h_count]*C_K))
                  # print(reward_k1)
                  # RL learn from this transition
                  # print(lr_k1, b_k1_done)
                 RL3.learn(observation_k1, action_k1, reward_k1, observation_k1_, lr_k3, b_k3_done, done1)
                
                 N_k5_hnsa[observation_k1, action_k1, h_count] = N_k5_hnsa[observation_k1, action_k1, h_count]+1
                 lr_k5 = (h_count_done+1)/(h_count_done+N_k5_hnsa[observation_k1, action_k1, h_count])
                 b_k5_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k5)/(N_k5_hnsa[observation_k1, action_k1, h_count]*C_K))
                  # print(reward_k1)
                  # RL learn from this transition
                  # print(lr_k1, b_k1_done)
                 RL5.learn(observation_k1, action_k1, reward_k1, observation_k1_, lr_k5, b_k5_done, done1)

            
            # if k2_done_label == 0:
            #      N_k2_hnsa[observation_k2, action_k2, h_count] = N_k2_hnsa[observation_k2, action_k2, h_count]+1
            #      lr_k2 = (h_count_done+1)/(h_count_done+N_k2_hnsa[observation_k2, action_k2, h_count])
            #      b_k2_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k2)/(N_k2_hnsa[observation_k2, action_k2, h_count]*C_K))
            #      # print(reward_k1)
            #      # RL learn from this transition
            #      # print(lr_k1, b_k1_done)
            #      RL2.learn(observation_k2, action_k2, reward_k2, observation_k2_, lr_k2, b_k2_done, done2)
            #      ep_reward_k2 += reward_k2
            #      observation_k2 = observation_k2_
            #      h_count_k2_done = h_count
                
            #      N_k4_hnsa[observation_k2, action_k2, h_count] = N_k4_hnsa[observation_k2, action_k2, h_count]+1
            #      lr_k4 = (h_count_done+1)/(h_count_done+N_k4_hnsa[observation_k2, action_k2, h_count])
            #      b_k4_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k4)/(N_k4_hnsa[observation_k2, action_k2, h_count]*C_K))
            #       # print(reward_k1)
            #       # RL learn from this transition
            #       # print(lr_k4, b_k4_done)
            #      RL4.learn(observation_k2, action_k2, reward_k2, observation_k2_, lr_k4, b_k4_done, done2)
                
            #      N_k6_hnsa[observation_k2, action_k2, h_count] = N_k6_hnsa[observation_k2, action_k2, h_count]+1
            #      lr_k6 = (h_count_done+1)/(h_count_done+N_k6_hnsa[observation_k2, action_k2, h_count])
            #      b_k6_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k6)/(N_k6_hnsa[observation_k2, action_k2, h_count]*C_K))
            #       # print(reward_k1)
            #       # RL learn from this transition
            #       # print(lr_k6, b_k6_done)
            #      RL6.learn(observation_k2, action_k2, reward_k2, observation_k2_, lr_k6, b_k6_done, done2)
                 # ep_reward_k1 += reward_k1
                 # observation_k2 = observation_k2_
                 # h_count_k1_done = h_count
     
            if done1:
                 k1_done_label = 1
                 # k1_done_label = 1
            # if done2:
            #      k2_done_label = 1
             # swap observation_k1
             # observation_k1 = observation_k12_%(xy_squre*xy_squre)
             # observation_k2 = int(observation_k12_/(xy_squre*xy_squre))
             # d_k1_destination_ = d_k1_destination
            
            
             # d_k1_destination_ = d_k1_destination
            # h_count=h_count+1
            if k3_done_label == 0:
             # observation_k12_= observation_k1_ + observation_k2_*xy_squre*xy_squre
                 N_k3_hnsa[observation_k3, action_k3, h_count] = N_k3_hnsa[observation_k3, action_k3, h_count]+1
                 lr_k3 = (h_count_done+1)/(h_count_done+N_k3_hnsa[observation_k3, action_k3, h_count])
                 b_k3_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k3)/(N_k3_hnsa[observation_k3, action_k3, h_count]*C_K))
                 # print(reward_k3)
                 # RL learn from this transition
                 # print(lr_k3, b_k3_done)
                 RL3.learn(observation_k3, action_k3, reward_k3, observation_k3_, lr_k3, b_k3_done, done3)
                 ep_reward_k3 += reward_k3
                 observation_k3 = observation_k3_
                 h_count_k3_done = h_count
                
                 N_k1_hnsa[observation_k3, action_k3, h_count] = N_k1_hnsa[observation_k3, action_k3, h_count]+1
                 lr_k1 = (h_count_done+1)/(h_count_done+N_k1_hnsa[observation_k3, action_k3, h_count])
                 b_k1_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k1)/(N_k1_hnsa[observation_k3, action_k3, h_count]*C_K))
                  # print(reward_k3)
                  # RL learn from this transition
                  # print(lr_k3, b_k3_done)
                 RL1.learn(observation_k3, action_k3, reward_k3, observation_k3_, lr_k1, b_k1_done, done3)
                  # ep_reward_k1 += reward_k2
                  # observation_k2 = observation_k2_
                  # h_count_k2_done = h_count
                 N_k5_hnsa[observation_k3, action_k3, h_count] = N_k5_hnsa[observation_k3, action_k3, h_count]+1
                 lr_k5 = (h_count_done+1)/(h_count_done+N_k5_hnsa[observation_k3, action_k3, h_count])
                 b_k5_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k5)/(N_k5_hnsa[observation_k3, action_k3, h_count]*C_K))
                  # print(reward_k3)
                  # RL learn from this transition
                  # print(lr_k3, b_k3_done)
                 RL5.learn(observation_k3, action_k3, reward_k3, observation_k3_, lr_k5, b_k5_done, done3)
            
            # if k4_done_label == 0:
            #      N_k4_hnsa[observation_k4, action_k4, h_count] = N_k4_hnsa[observation_k4, action_k4, h_count]+1
            #      lr_k4 = (h_count_done+1)/(h_count_done+N_k4_hnsa[observation_k4, action_k4, h_count])
            #      b_k4_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k4)/(N_k4_hnsa[observation_k4, action_k4, h_count]*C_K))
            #      # print(reward_k1)
            #      # RL learn from this transition
            #      # print(lr_k1, b_k1_done)
            #      RL4.learn(observation_k4, action_k4, reward_k4, observation_k4_, lr_k4, b_k4_done, done4)
            #      ep_reward_k4 += reward_k4
            #      observation_k4 = observation_k4_
            #      h_count_k4_done = h_count
                
            #      N_k2_hnsa[observation_k4, action_k4, h_count] = N_k2_hnsa[observation_k4, action_k4, h_count]+1
            #      lr_k2 = (h_count_done+1)/(h_count_done+N_k2_hnsa[observation_k4, action_k4, h_count])
            #      b_k2_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k2)/(N_k2_hnsa[observation_k4, action_k4, h_count]*C_K))
            #       # print(reward_k2)
            #       # RL learn from this transition
            #       # print(lr_k2, b_k2_done)
            #      RL2.learn(observation_k4, action_k4, reward_k4, observation_k4_, lr_k2, b_k2_done, done4)
            #       # ep_reward_k1 += reward_k1
            #       # observation_k4 = observation_k4_
            #       # h_count_k1_done = h_count
            #      N_k6_hnsa[observation_k4, action_k4, h_count] = N_k6_hnsa[observation_k4, action_k4, h_count]+1
            #      lr_k6 = (h_count_done+1)/(h_count_done+N_k6_hnsa[observation_k4, action_k4, h_count])
            #      b_k6_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k6)/(N_k6_hnsa[observation_k4, action_k4, h_count]*C_K))
            #       # print(reward_k6)
            #       # RL learn from this transition
            #       # print(lr_k6, b_k6_done)
            #      RL6.learn(observation_k4, action_k4, reward_k4, observation_k4_, lr_k6, b_k6_done, done4)
     
            if done3:
                 k3_done_label = 1
                 # k1_done_label = 1
            # if done4:
            #      k4_done_label = 1
             # swap observation_k1
             # observation_k1 = observation_k12_%(xy_squre*xy_squre)
             # observation_k2 = int(observation_k12_/(xy_squre*xy_squre))
             # d_k1_destination_ = d_k1_destination
            
            
             # d_k1_destination_ = d_k1_destination
            # h_count=h_count+1 
            if k5_done_label == 0:
            # observation_k12_= observation_k1_ + observation_k2_*xy_squre*xy_squre
                N_k5_hnsa[observation_k5, action_k5, h_count] = N_k5_hnsa[observation_k5, action_k5, h_count]+1
                lr_k5 = (h_count_done+1)/(h_count_done+N_k5_hnsa[observation_k5, action_k5, h_count])
                b_k5_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k5)/(N_k5_hnsa[observation_k5, action_k5, h_count]*C_K))
                RL5.learn(observation_k5, action_k5, reward_k5, observation_k5_, lr_k5, b_k5_done, done5)
                ep_reward_k5 += reward_k5
                observation_k5 = observation_k5_
                h_count_k5_done = h_count
                
                N_k1_hnsa[observation_k5, action_k5, h_count] = N_k1_hnsa[observation_k5, action_k5, h_count]+1
                lr_k1 = (h_count_done+1)/(h_count_done+N_k1_hnsa[observation_k5, action_k5, h_count])
                b_k1_done = 0.000001*math.sqrt(h_count_done**5*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k1)/(N_k1_hnsa[observation_k5, action_k5, h_count]*C_K))
                # print(reward_k5)
                # RL learn from this transition
                # print(lr_k5, b_k5_done)
                RL1.learn(observation_k5, action_k5, reward_k5, observation_k5_, lr_k1, b_k1_done, done5)
                # ep_reward_k1 += reward_k2
                # observation_k2 = observation_k2_
                # h_count_k2_done = h_count
                N_k3_hnsa[observation_k5, action_k5, h_count] = N_k3_hnsa[observation_k5, action_k5, h_count]+1
                lr_k3 = (h_count_done+1)/(h_count_done+N_k3_hnsa[observation_k5, action_k5, h_count])
                b_k3_done = 0.000001*math.sqrt(h_count_done**5*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k3)/(N_k3_hnsa[observation_k5, action_k5, h_count]*C_K))
                RL3.learn(observation_k5, action_k5, reward_k5, observation_k5_, lr_k3, b_k3_done, done5)
            
            # if k6_done_label == 0:
            #     N_k6_hnsa[observation_k6, action_k6, h_count] = N_k6_hnsa[observation_k6, action_k6, h_count]+1
            #     lr_k6 = (h_count_done+1)/(h_count_done+N_k6_hnsa[observation_k6, action_k6, h_count])
            #     b_k6_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k6)/(N_k6_hnsa[observation_k6, action_k6, h_count]*C_K))
            #     # print(reward_k1)
            #     # RL learn from this transition
            #     # print(lr_k1, b_k1_done)
            #     RL6.learn(observation_k6, action_k6, reward_k6, observation_k6_, lr_k6, b_k6_done, done6)
            #     ep_reward_k6 += reward_k6
            #     observation_k6 = observation_k6_
            #     h_count_k6_done = h_count
                
            #     N_k2_hnsa[observation_k6, action_k6, h_count] = N_k2_hnsa[observation_k6, action_k6, h_count]+1
            #     lr_k2 = (h_count_done+1)/(h_count_done+N_k2_hnsa[observation_k6, action_k6, h_count])
            #     b_k2_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k2)/(N_k2_hnsa[observation_k6, action_k6, h_count]*C_K))
            #     # print(reward_k2)
            #     # RL learn from this transition
            #     # print(lr_k2, b_k2_done)
            #     RL2.learn(observation_k6, action_k6, reward_k6, observation_k6_, lr_k2, b_k2_done, done6)
                
            #     N_k4_hnsa[observation_k6, action_k6, h_count] = N_k4_hnsa[observation_k6, action_k6, h_count]+1
            #     lr_k4 = (h_count_done+1)/(h_count_done+N_k4_hnsa[observation_k6, action_k6, h_count])
            #     b_k4_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k4)/(N_k4_hnsa[observation_k6, action_k6, h_count]*C_K))
            #     # print(reward_k6)
            #     # RL learn from this transition
            #     # print(lr_k6, b_k6_done)
            #     RL4.learn(observation_k6, action_k6, reward_k6, observation_k6_, lr_k4, b_k4_done, done6)
     
            if done5:
                k5_done_label = 1
                # k1_done_label = 1
            # if done6:
            #     k6_done_label = 1
            # ep_reward_k12 += reward_k12
            # break while loop when end of this episode
            if k1_done_label == 1:
                # if k2_done_label == 1:
                    if k3_done_label == 1:
                        # if k4_done_label == 1:
                            if k5_done_label == 1:
                                # if k6_done_label == 1:
                                    break
        # print('1',episode, h_count_k1_done+1, h_count_k3_done+1)
        # print('2',episode, h_count_k3_done+1, h_count_k4_done+1)
        # print('3',episode, h_count_k5_done+1, h_count_k6_done+1)
        # print('**********************************************************')##    print(episode, h_count, ep_reward_k2+ep_reward_k1+(h_count_done-ep_reward_k1)*reward_k1+(h_count_done-ep_reward_k2)*reward_k2)
        R_sum =(ep_R_k1/(h_count_k1_done+1)+ep_R_k3/(h_count_k3_done+1)+ep_R_k5/(h_count_k5_done+1))/3
        filename = 'ep_H'+str(P_SN_mav)+'.txt'
        with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
              fileobject.write(str(h_count)+'\n')  
        # ep_sum2 = (ep_reward_k1+(h_count_done-h_count_k1_done-1)*1+ep_reward_k2+(h_count_done-h_count_k2_done-1)*1+ep_reward_k3+(h_count_done-h_count_k3_done-1)*1+ep_reward_k4+(h_count_done-h_count_k4_done-1)*1)/2
        filename = 'ep_reward'+str(P_SN_mav)+'.txt'
        with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
              fileobject.write(str(R_sum)+'\n')  
    # end of game
print('game over')
env.destroy()