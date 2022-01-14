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
import seaborn as sns
import matplotlib.pyplot as plt 
import turtle as tu
env = gym.make('LargeGridWorld-v0')
env = env.unwrapped
env2 = gym.make('LargeGridWorld-v0')
env2 = env2.unwrapped
env10 = gym.make('LargeGridWorld-v0')
env10 = env10.unwrapped
# env4 = gym.make('LargeGridWorld-v0')
# env4 = env4.unwrapped
# env5 = gym.make('LargeGridWorld-v0')
# env5 = env5.unwrapped
# env6 = gym.make('LargeGridWorld-v0')
# env6 = env6.unwrapped
env7 = gym.make('LargeGridWorld-v0')
env7 = env7.unwrapped
# env8 = gym.make('LargeGridWorld-v0')
# env8 = env8.unwrapped
# env9 = gym.make('LargeGridWorld-v0')
# env9 = env9.unwrapped
import matplotlib.patches as pc
#env = Maze()
import os
import matplotlib;matplotlib.use('Agg')


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
# 		print ("---  OK  ---")
 
	else:
		return
		
    #调用函数




np.random.seed(0)
fig = plt.figure()
C_K = 1
K = 4
M_k = 1
Antenna_L = 4
ACTION_SHAPE = 5*20
P_SN_mav_array = [0.19952623149688797]
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
pho_k2 = 0.01
pho_k3 = 0.01
pho_k4 = 0.01
pho_k5 = 0.01
pho_k6 = 0.01
pho_k7 = 0.01
pho_k8 = 0.01
pho_k10 = 0.01

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
    RL2 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL3 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL4 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL5 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL6 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    RL7 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL8 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    # RL9 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    RL10 = QLearningTable(actions=list(range(ACTION_SHAPE)))
    
    N_k1_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k2_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k3_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k4_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k5_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k6_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k7_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k8_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k10_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    
    
    #print(RL.q_table)
    # d_k1_destination_max = (math.sqrt((xy_squre)*(xy_squre)*2))*Delta_xy
    
    for episode in range(n_count_done):
        # initial observation_k1
        ep_R_k1 = 0
        ep_R_k2 = 0
        ep_R_k3 = 0
        ep_R_k4 = 0
        ep_R_k5 = 0
        ep_R_k6 = 0
        ep_R_k7 = 0
        ep_R_k8 = 0
        ep_R_k10 = 0
        ep_reward_k1=0
        observation_k1, [x_k1_destination, y_k1_destination] = env.reset()
        # print(observation_k1, [x_k1_destination, y_k1_destination])
        x_1 = observation_k1%xy_squre
        y_1 = int(observation_k1/xy_squre)
        d_k1_destination_max = (math.sqrt((x_1-x_k1_destination)*(x_1-x_k1_destination)+(y_1-y_k1_destination)*(y_1-y_k1_destination)))*Delta_xy
        d_k1_destination_ = (math.sqrt((x_1-x_k1_destination)*(x_1-x_k1_destination)+(y_1-y_k1_destination)*(y_1-y_k1_destination)))*Delta_xy
        label_done = 0
        # ep_reward_k12=0
        ep_reward_k2=0
        observation_k2, [x_k2_destination, y_k2_destination] = env2.reset()
        # print(observation_k2, [x_k2_destination, y_k2_destination])
        x_2 = observation_k2%xy_squre
        y_2 = int(observation_k2/xy_squre)
        d_k2_destination_max = (math.sqrt((x_2-x_k2_destination)*(x_2-x_k2_destination)+(y_2-y_k2_destination)*(y_2-y_k2_destination)))*Delta_xy
        d_k2_destination_ = (math.sqrt((x_2-x_k2_destination)*(x_2-x_k2_destination)+(y_2-y_k2_destination)*(y_2-y_k2_destination)))*Delta_xy
        # label_done = 0
        
        k1_done_label = 0
        k2_done_label = 0
        h_count_k1_done = h_count_done
        h_count_k2_done = h_count_done
        
        
        # ep_reward_k3=0
        # observation_k3, [x_k3_destination, y_k3_destination] = env3.reset()
        # # print(observation_k3, [x_k3_destination, y_k3_destination])
        # x_3 = observation_k3%xy_squre
        # y_3 = int(observation_k3/xy_squre)
        # d_k3_destination_max = (math.sqrt((x_3-x_k3_destination)*(x_3-x_k3_destination)+(y_3-y_k3_destination)*(y_3-y_k3_destination)))*Delta_xy
        # d_k3_destination_ = (math.sqrt((x_3-x_k3_destination)*(x_3-x_k3_destination)+(y_3-y_k3_destination)*(y_3-y_k3_destination)))*Delta_xy
        # label_done = 0
        # ep_reward_k4=0
        # observation_k4, [x_k4_destination, y_k4_destination] = env4.reset()
        # # print(observation_k4, [x_k4_destination, y_k4_destination])
        # x_4 = observation_k4%xy_squre
        # y_4 = int(observation_k4/xy_squre)
        # d_k4_destination_max = (math.sqrt((x_4-x_k4_destination)*(x_4-x_k4_destination)+(y_4-y_k4_destination)*(y_4-y_k4_destination)))*Delta_xy
        # d_k4_destination_ = (math.sqrt((x_4-x_k4_destination)*(x_4-x_k4_destination)+(y_4-y_k4_destination)*(y_4-y_k4_destination)))*Delta_xy
        # # label_done = 0
        
        # k3_done_label = 0
        # k4_done_label = 0
        # h_count_k3_done = h_count_done
        # h_count_k4_done = h_count_done
        
        # ep_reward_k5=0
        # observation_k5, [x_k5_destination, y_k5_destination] = env5.reset()
        # x_5 = observation_k5%xy_squre
        # y_5 = int(observation_k5/xy_squre)
        # d_k5_destination_max = (math.sqrt((x_5-x_k5_destination)*(x_5-x_k5_destination)+(y_5-y_k5_destination)*(y_5-y_k5_destination)))*Delta_xy
        # d_k5_destination_ = (math.sqrt((x_5-x_k5_destination)*(x_5-x_k5_destination)+(y_5-y_k5_destination)*(y_5-y_k5_destination)))*Delta_xy
        # # label_done = 0
        # ep_reward_k6=0
        # observation_k6, [x_k6_destination, y_k6_destination] = env6.reset()
        # x_6 = observation_k6%xy_squre
        # y_6 = int(observation_k6/xy_squre)
        # d_k6_destination_max = (math.sqrt((x_6-x_k6_destination)*(x_6-x_k6_destination)+(y_6-y_k6_destination)*(y_6-y_k6_destination)))*Delta_xy
        # d_k6_destination_ = (math.sqrt((x_6-x_k6_destination)*(x_6-x_k6_destination)+(y_6-y_k6_destination)*(y_6-y_k6_destination)))*Delta_xy
        # # label_done = 0
        
        # k5_done_label = 0
        # k6_done_label = 0
        # h_count_k5_done = h_count_done
        # h_count_k6_done = h_count_done
        
        
        ep_reward_k7=0
        observation_k7, [x_k7_destination, y_k7_destination] = env7.reset()
        x_7 = observation_k7%xy_squre
        y_7 = int(observation_k7/xy_squre)
        d_k7_destination_max = (math.sqrt((x_7-x_k7_destination)*(x_7-x_k7_destination)+(y_7-y_k7_destination)*(y_7-y_k7_destination)))*Delta_xy
        d_k7_destination_ = (math.sqrt((x_7-x_k7_destination)*(x_7-x_k7_destination)+(y_7-y_k7_destination)*(y_7-y_k7_destination)))*Delta_xy
        # label_done = 0
        # ep_reward_k8=0
        # observation_k8, [x_k8_destination, y_k8_destination] = env8.reset()
        # x_8 = observation_k8%xy_squre
        # y_8 = int(observation_k8/xy_squre)
        # d_k8_destination_max = (math.sqrt((x_8-x_k8_destination)*(x_8-x_k8_destination)+(y_8-y_k8_destination)*(y_8-y_k8_destination)))*Delta_xy
        # d_k8_destination_ = (math.sqrt((x_8-x_k8_destination)*(x_8-x_k8_destination)+(y_8-y_k8_destination)*(y_8-y_k8_destination)))*Delta_xy
        # # label_done = 0
        
        k7_done_label = 0
        # k8_done_label = 0
        h_count_k7_done = h_count_done
        # h_count_k8_done = h_count_done
        
        # ep_reward_k9=0
        # observation_k9, [x_k9_destination, y_k9_destination] = env9.reset()
        # x_9 = observation_k9%xy_squre
        # y_9 = int(observation_k9/xy_squre)
        # d_k9_destination_max = (math.sqrt((x_9-x_k9_destination)*(x_9-x_k9_destination)+(y_9-y_k9_destination)*(y_9-y_k9_destination)))*Delta_xy
        # d_k9_destination_ = (math.sqrt((x_9-x_k9_destination)*(x_9-x_k9_destination)+(y_9-y_k9_destination)*(y_9-y_k9_destination)))*Delta_xy
        
        # k9_done_label = 0
        # h_count_k9_done = h_count_done
        
        # h_count_done = 8000
        
        ep_reward_k10=0
        observation_k10, [x_k10_destination, y_k10_destination] = env10.reset()
        x_10 = observation_k10%xy_squre
        y_10 = int(observation_k10/xy_squre)
        d_k10_destination_max = (math.sqrt((x_10-x_k10_destination)*(x_10-x_k10_destination)+(y_10-y_k10_destination)*(y_10-y_k10_destination)))*Delta_xy
        d_k10_destination_ = (math.sqrt((x_10-x_k10_destination)*(x_10-x_k10_destination)+(y_10-y_k10_destination)*(y_10-y_k10_destination)))*Delta_xy
        k10_done_label = 0
        h_count_k10_done = h_count_done
        x_k1_array = []
        y_k1_array = []
        x_k2_array = []
        y_k2_array = []
        x_k7_array = []
        y_k7_array = []
        x_k10_array = []
        y_k10_array = []
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
            [x_SN7, y_SN7] = [0, 15]
            [x_SN8, y_SN8] = [5, 17]
            [x_SN9, y_SN9] = [15, 17]
            [x_SN10, y_SN10] = [15, 17]
            [x_SN11, y_SN11] = [5, 3]
            [x_SN12, y_SN12] = [10, 5]
            [x_SN13, y_SN13] = [15, 5]
            
            d_k1_SN1 = (math.sqrt((x_1-x_SN1)*(x_1-x_SN1)+(y_1-y_SN1)*(y_1-y_SN1)))*Delta_xy+1
            d_k1_SN2 = (math.sqrt((x_1-x_SN2)*(x_1-x_SN2)+(y_1-y_SN2)*(y_1-y_SN2)))*Delta_xy+1
            d_k1_SN3 = (math.sqrt((x_1-x_SN3)*(x_1-x_SN3)+(y_1-y_SN3)*(y_1-y_SN3)))*Delta_xy+1
            d_k1_SN4 = (math.sqrt((x_1-x_SN4-xy_squre)*(x_1-x_SN4-xy_squre)+(y_1-y_SN4)*(y_1-y_SN4)))*Delta_xy+1
            d_k1_SN5 = (math.sqrt((x_1-x_SN5-xy_squre)*(x_1-x_SN5-xy_squre)+(y_1-y_SN5)*(y_1-y_SN5)))*Delta_xy+1
            d_k1_SN6 = (math.sqrt((x_1-x_SN6-xy_squre)*(x_1-x_SN6-xy_squre)+(y_1-y_SN6)*(y_1-y_SN6)))*Delta_xy+1  
            d_k1_destination = (math.sqrt((x_1-x_k1_destination)*(x_1-x_k1_destination)+(y_1-y_k1_destination)*(y_1-y_k1_destination)))*Delta_xy
            beta_k1_SN1 = math.pow(10, -3)*math.pow(d_k1_SN1, -2.2)
            beta_k1_SN2 = math.pow(10, -3)*math.pow(d_k1_SN2, -2.2)
            beta_k1_SN3 = math.pow(10, -3)*math.pow(d_k1_SN3, -2.2) 
            beta_k1_SN4 = math.pow(10, -3)*math.pow(d_k1_SN4, -2.2)
            beta_k1_SN5 = math.pow(10, -3)*math.pow(d_k1_SN5, -2.2)
            beta_k1_SN6 = math.pow(10, -3)*math.pow(d_k1_SN6, -2.2) 
            
            theta_k1_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_1)/d_k1_SN1))
            theta_k1_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_1)/d_k1_SN2))
            theta_k1_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_1)/d_k1_SN3))
            theta_k1_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4+xy_squre-x_1)/d_k1_SN4))
            theta_k1_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5+xy_squre-x_1)/d_k1_SN5))
            theta_k1_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6+xy_squre-x_1)/d_k1_SN6))
            
            g_k1_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN1), math.exp(1)**(-1*1j*theta_k1_SN1*2), math.exp(1)**(-1*1j*theta_k1_SN1*3)])
            g_k1_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN2), math.exp(1)**(-1*1j*theta_k1_SN2*2), math.exp(1)**(-1*1j*theta_k1_SN2*3)])
            g_k1_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN3), math.exp(1)**(-1*1j*theta_k1_SN3*2), math.exp(1)**(-1*1j*theta_k1_SN3*3)])
            g_k1_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN4), math.exp(1)**(-1*1j*theta_k1_SN4*2), math.exp(1)**(-1*1j*theta_k1_SN4*3)])
            g_k1_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN5), math.exp(1)**(-1*1j*theta_k1_SN5*2), math.exp(1)**(-1*1j*theta_k1_SN5*3)])
            g_k1_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN6), math.exp(1)**(-1*1j*theta_k1_SN6*2), math.exp(1)**(-1*1j*theta_k1_SN6*3)])
            
            g_k1_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN1 = g_k1_SN1_LOS + g_k1_SN1_NLOS
            g_k1_SN2 = g_k1_SN2_LOS + g_k1_SN2_NLOS
            g_k1_SN3 = g_k1_SN3_LOS + g_k1_SN3_NLOS
            g_k1_SN4 = g_k1_SN4_LOS + g_k1_SN4_NLOS
            g_k1_SN5 = g_k1_SN5_LOS + g_k1_SN5_NLOS
            g_k1_SN6 = g_k1_SN6_LOS + g_k1_SN6_NLOS
            h_k1_SN1 = math.sqrt(beta_k1_SN1)*g_k1_SN1
            h_k1_SN2 = math.sqrt(beta_k1_SN2)*g_k1_SN2
            h_k1_SN3 = math.sqrt(beta_k1_SN3)*g_k1_SN3
            h_k1_SN4 = math.sqrt(beta_k1_SN4)*g_k1_SN4            
            h_k1_SN5 = math.sqrt(beta_k1_SN5)*g_k1_SN5
            h_k1_SN6 = math.sqrt(beta_k1_SN6)*g_k1_SN6
            #communication model
            H_k1_PART1 = np.array([h_k1_SN1[0],h_k1_SN1[1],h_k1_SN1[2],h_k1_SN1[3],h_k1_SN2[0],h_k1_SN2[1],h_k1_SN2[2],h_k1_SN2[3],h_k1_SN3[0],h_k1_SN3[1],h_k1_SN3[2],h_k1_SN3[3]]).T
            H_k1_PART2 = np.array([h_k1_SN4[0],h_k1_SN4[1],h_k1_SN4[2],h_k1_SN4[3],h_k1_SN5[0],h_k1_SN5[1],h_k1_SN5[2],h_k1_SN5[3],h_k1_SN6[0],h_k1_SN6[1],h_k1_SN6[2],h_k1_SN6[3]]).T
    
            d_k1_SN7 = (math.sqrt((x_1-x_SN7-xy_squre)*(x_1-x_SN7-xy_squre)+(y_1-y_SN7-xy_squre)*(y_1-y_SN7-xy_squre)))*Delta_xy+1
            d_k1_SN8 = (math.sqrt((x_1-x_SN8-xy_squre)*(x_1-x_SN8-xy_squre)+(y_1-y_SN8-xy_squre)*(y_1-y_SN8-xy_squre)))*Delta_xy+1
            d_k1_SN9 = (math.sqrt((x_1-x_SN9-xy_squre)*(x_1-x_SN9-xy_squre)+(y_1-y_SN9-xy_squre)*(y_1-y_SN9-xy_squre)))*Delta_xy+1
            beta_k1_SN7 = math.pow(10, -3)*math.pow(d_k1_SN7, -2.2)
            beta_k1_SN8 = math.pow(10, -3)*math.pow(d_k1_SN8, -2.2)
            beta_k1_SN9 = math.pow(10, -3)*math.pow(d_k1_SN9, -2.2) 
            theta_k1_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_1)/d_k1_SN7))
            theta_k1_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_1)/d_k1_SN8))
            theta_k1_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_1)/d_k1_SN9))
            g_k1_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN7), math.exp(1)**(-1*1j*theta_k1_SN7*2), math.exp(1)**(-1*1j*theta_k1_SN7*3)])
            g_k1_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN8), math.exp(1)**(-1*1j*theta_k1_SN8*2), math.exp(1)**(-1*1j*theta_k1_SN8*3)])
            g_k1_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN9), math.exp(1)**(-1*1j*theta_k1_SN9*2), math.exp(1)**(-1*1j*theta_k1_SN9*3)])
            g_k1_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN7 = g_k1_SN7_LOS + g_k1_SN7_NLOS
            g_k1_SN8 = g_k1_SN8_LOS + g_k1_SN8_NLOS
            g_k1_SN9 = g_k1_SN9_LOS + g_k1_SN9_NLOS
            h_k1_SN7 = math.sqrt(beta_k1_SN7)*g_k1_SN7            
            h_k1_SN8 = math.sqrt(beta_k1_SN8)*g_k1_SN8
            h_k1_SN9 = math.sqrt(beta_k1_SN9)*g_k1_SN9
            H_k1_PART3 = np.array([h_k1_SN7[0],h_k1_SN7[1],h_k1_SN7[2],h_k1_SN7[3],h_k1_SN8[0],h_k1_SN8[1],h_k1_SN8[2],h_k1_SN8[3],h_k1_SN9[0],h_k1_SN9[1],h_k1_SN9[2],h_k1_SN9[3]]).T

            
            d_k1_SN10 = (math.sqrt((x_1-x_SN10)*(x_1-x_SN10)+(y_1-y_SN10-xy_squre)*(y_1-y_SN10-xy_squre)))*Delta_xy+1
            d_k1_SN11 = (math.sqrt((x_1-x_SN11)*(x_1-x_SN11)+(y_1-y_SN11-xy_squre)*(y_1-y_SN11-xy_squre)))*Delta_xy+1
            d_k1_SN12 = (math.sqrt((x_1-x_SN12)*(x_1-x_SN12)+(y_1-y_SN12-xy_squre)*(y_1-y_SN12-xy_squre)))*Delta_xy+1
            beta_k1_SN10 = math.pow(10, -3)*math.pow(d_k1_SN10, -2.2)
            beta_k1_SN11 = math.pow(10, -3)*math.pow(d_k1_SN11, -2.2)
            beta_k1_SN12 = math.pow(10, -3)*math.pow(d_k1_SN12, -2.2) 
            theta_k1_SN10 = math.exp(1)**(-1*1j*(math.pi*(x_SN10-x_1)/d_k1_SN10))
            theta_k1_SN11 = math.exp(1)**(-1*1j*(math.pi*(x_SN11-x_1)/d_k1_SN11))
            theta_k1_SN12 = math.exp(1)**(-1*1j*(math.pi*(x_SN12-x_1)/d_k1_SN12))
            g_k1_SN10_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN10), math.exp(1)**(-1*1j*theta_k1_SN10*2), math.exp(1)**(-1*1j*theta_k1_SN10*3)])
            g_k1_SN11_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN11), math.exp(1)**(-1*1j*theta_k1_SN11*2), math.exp(1)**(-1*1j*theta_k1_SN11*3)])
            g_k1_SN12_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN12), math.exp(1)**(-1*1j*theta_k1_SN12*2), math.exp(1)**(-1*1j*theta_k1_SN12*3)])
            g_k1_SN10_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN11_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN12_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k1_SN10 = g_k1_SN10_LOS + g_k1_SN10_NLOS
            g_k1_SN11 = g_k1_SN11_LOS + g_k1_SN11_NLOS
            g_k1_SN12 = g_k1_SN12_LOS + g_k1_SN12_NLOS
            h_k1_SN10 = math.sqrt(beta_k1_SN10)*g_k1_SN10            
            h_k1_SN11 = math.sqrt(beta_k1_SN11)*g_k1_SN11
            h_k1_SN12 = math.sqrt(beta_k1_SN12)*g_k1_SN12
            H_k1_PART4 = np.array([h_k1_SN10[0],h_k1_SN10[1],h_k1_SN10[2],h_k1_SN10[3],h_k1_SN11[0],h_k1_SN11[1],h_k1_SN11[2],h_k1_SN11[3],h_k1_SN12[0],h_k1_SN12[1],h_k1_SN12[2],h_k1_SN12[3]]).T
            
            #communication model
            H_k1 = np.array([H_k1_PART1, H_k1_PART2, H_k1_PART3, H_k1_PART4]).T
            # try:
            #     W_bar_k1 = H_k1.dot(lg.inv(H_k1.conj().T.dot(H_k1)))
            # except:
            #     W_bar_k1 = W_bar_k1
            # else:
            #     W_bar_k1 = H_k1.dot(lg.inv(H_k1.conj().T.dot(H_k1)))
            W_bar_k1 = H_k1.dot(lg.inv(H_k1.conj().T.dot(H_k1)))
            w_k1_k1 = W_bar_k1[:, 0]/(np.linalg.norm(W_bar_k1[:, 0]))
            # w_k1_k2 = W_bar_k1[:, 1]/(np.linalg.norm(W_bar_k1[:, 1]))
            # w_k1_k3 = W_bar_k1[:, 2]/(np.linalg.norm(W_bar_k1[:, 2]))
            # w_k1_k4 = W_bar_k1[:, ]/(np.linalg.norm(W_bar_k1[:, 2]))
            # print(np.linalg.norm(W_bar_k1[:, 0]),np.linalg.norm(W_bar_k1[:, 1]),np.linalg.norm(W_bar_k1[:, 2]))
            # w_k1_SN3 = W_bar_k1[:, 2]/(np.linalg.norm(W_bar_k1[:, 2]))
            # w_k1_SN4 = W_bar_k1[:, 3]/(np.linalg.norm(W_bar_k1[:, 3]))
            # w_k1_SN5 = W_bar_k1[:, 4]/(np.linalg.norm(W_bar_k1[:, 4]))
            # w_k1_SN6 = W_bar_k1[:, 5]/(np.linalg.norm(W_bar_k1[:, 5]))
            # w_check1 = np.linalg.norm(w_k1_SN1)
            # w_check2 = np.linalg.norm(w_k1_SN2)
            # w_check3 = np.linalg.norm(w_k1_SN3)
            # w_k1_SN9=w_k1_k3[6:9]
            # w_k1_SN8=w_k1_k3[3:6]
            # w_k1_SN7=w_k1_k3[0:3]
            # w_k1_SN6=w_k1_k2[6:9]
            # w_k1_SN5=w_k1_k2[3:6]
            # w_k1_SN4=w_k1_k2[0:3]
            w_k1_SN3=w_k1_k1[8:12]
            w_k1_SN2=w_k1_k1[4:8]
            w_k1_SN1=w_k1_k1[0:4]
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
    
            x_2 = observation_k2%xy_squre
            y_2 = int(observation_k2/xy_squre)
            d_k2_SN4 = (math.sqrt((x_2-x_SN4)*(x_2-x_SN4)+(y_2-y_SN4)*(y_2-y_SN4)))*Delta_xy+1
            d_k2_SN5 = (math.sqrt((x_2-x_SN5)*(x_2-x_SN5)+(y_2-y_SN5)*(y_2-y_SN5)))*Delta_xy+1
            d_k2_SN6 = (math.sqrt((x_2-x_SN6)*(x_2-x_SN6)+(y_2-y_SN6)*(y_2-y_SN6)))*Delta_xy+1
            d_k2_SN1 = (math.sqrt((x_2+xy_squre-x_SN1)*(x_2+xy_squre-x_SN1)+(y_2-y_SN1)*(y_2-y_SN1)))*Delta_xy+1
            d_k2_SN2 = (math.sqrt((x_2+xy_squre-x_SN2)*(x_2+xy_squre-x_SN2)+(y_2-y_SN2)*(y_2-y_SN2)))*Delta_xy+1
            d_k2_SN3 = (math.sqrt((x_2+xy_squre-x_SN3)*(x_2+xy_squre-x_SN3)+(y_2-y_SN3)*(y_2-y_SN3)))*Delta_xy+1
            d_k2_destination = (math.sqrt((x_2-x_k2_destination)*(x_2-x_k2_destination)+(y_2-y_k2_destination)*(y_2-y_k2_destination)))*Delta_xy
            beta_k2_SN4 = math.pow(10, -3)*math.pow(d_k2_SN4, -2.2)
            beta_k2_SN5 = math.pow(10, -3)*math.pow(d_k2_SN5, -2.2)
            beta_k2_SN6 = math.pow(10, -3)*math.pow(d_k2_SN6, -2.2)
            beta_k2_SN1 = math.pow(10, -3)*math.pow(d_k2_SN1, -2.2)
            beta_k2_SN2 = math.pow(10, -3)*math.pow(d_k2_SN2, -2.2)
            beta_k2_SN3 = math.pow(10, -3)*math.pow(d_k2_SN3, -2.2)
            theta_k2_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_2)/d_k2_SN4))
            theta_k2_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_2)/d_k2_SN5))
            theta_k2_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_2)/d_k2_SN6))
            theta_k2_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_2-xy_squre)/d_k2_SN1))
            theta_k2_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_2-xy_squre)/d_k2_SN2))
            theta_k2_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_2-xy_squre)/d_k2_SN3))
            g_k2_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN4), math.exp(1)**(-1*1j*theta_k2_SN4*2), math.exp(1)**(-1*1j*theta_k2_SN4*3)])
            g_k2_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN5), math.exp(1)**(-1*1j*theta_k2_SN5*2), math.exp(1)**(-1*1j*theta_k2_SN5*3)])
            g_k2_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN6), math.exp(1)**(-1*1j*theta_k2_SN6*2), math.exp(1)**(-1*1j*theta_k2_SN6*3)])
            g_k2_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN1), math.exp(1)**(-1*1j*theta_k2_SN1*2), math.exp(1)**(-1*1j*theta_k2_SN1*3)])
            g_k2_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN2), math.exp(1)**(-1*1j*theta_k2_SN2*2), math.exp(1)**(-1*1j*theta_k2_SN2*3)])
            g_k2_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN3), math.exp(1)**(-1*1j*theta_k2_SN3*2), math.exp(1)**(-1*1j*theta_k2_SN3*3)])
            g_k2_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN4 = g_k2_SN4_LOS + g_k2_SN4_NLOS
            g_k2_SN5 = g_k2_SN5_LOS + g_k2_SN5_NLOS
            g_k2_SN6 = g_k2_SN6_LOS + g_k2_SN6_NLOS
            g_k2_SN1 = g_k2_SN1_LOS + g_k2_SN1_NLOS
            g_k2_SN2 = g_k2_SN2_LOS + g_k2_SN2_NLOS
            g_k2_SN3 = g_k2_SN3_LOS + g_k2_SN3_NLOS
            h_k2_SN4 = math.sqrt(beta_k2_SN4)*g_k2_SN4            
            h_k2_SN5 = math.sqrt(beta_k2_SN5)*g_k2_SN5
            h_k2_SN6 = math.sqrt(beta_k2_SN6)*g_k2_SN6
            h_k2_SN1 = math.sqrt(beta_k2_SN1)*g_k2_SN1            
            h_k2_SN2 = math.sqrt(beta_k2_SN2)*g_k2_SN2
            h_k2_SN3 = math.sqrt(beta_k2_SN3)*g_k2_SN3
            
            d_k2_SN7 = (math.sqrt((x_2-x_SN7)*(x_2-x_SN7)+(y_2-y_SN7-xy_squre)*(y_2-y_SN7-xy_squre)))*Delta_xy+1
            d_k2_SN8 = (math.sqrt((x_2-x_SN8)*(x_2-x_SN8)+(y_2-y_SN8-xy_squre)*(y_2-y_SN8-xy_squre)))*Delta_xy+1
            d_k2_SN9 = (math.sqrt((x_2-x_SN9)*(x_2-x_SN9)+(y_2-y_SN9-xy_squre)*(y_2-y_SN9-xy_squre)))*Delta_xy+1
            beta_k2_SN7 = math.pow(10, -3)*math.pow(d_k2_SN7, -2.2)
            beta_k2_SN8 = math.pow(10, -3)*math.pow(d_k2_SN8, -2.2)
            beta_k2_SN9 = math.pow(10, -3)*math.pow(d_k2_SN9, -2.2) 
            theta_k2_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_2)/d_k2_SN7))
            theta_k2_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_2)/d_k2_SN8))
            theta_k2_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_2)/d_k2_SN9))
            g_k2_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN7), math.exp(1)**(-1*1j*theta_k2_SN7*2), math.exp(1)**(-1*1j*theta_k2_SN7*3)])
            g_k2_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN8), math.exp(1)**(-1*1j*theta_k2_SN8*2), math.exp(1)**(-1*1j*theta_k2_SN8*3)])
            g_k2_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN9), math.exp(1)**(-1*1j*theta_k2_SN9*2), math.exp(1)**(-1*1j*theta_k2_SN9*3)])
            g_k2_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN7 = g_k2_SN7_LOS + g_k2_SN7_NLOS
            g_k2_SN8 = g_k2_SN8_LOS + g_k2_SN8_NLOS
            g_k2_SN9 = g_k2_SN9_LOS + g_k2_SN9_NLOS
            h_k2_SN7 = math.sqrt(beta_k2_SN7)*g_k2_SN7            
            h_k2_SN8 = math.sqrt(beta_k2_SN8)*g_k2_SN8
            h_k2_SN9 = math.sqrt(beta_k2_SN9)*g_k2_SN9
            
            d_k2_SN10 = (math.sqrt((x_2+xy_squre-x_SN10)*(x_2+xy_squre-x_SN10)+(y_2-y_SN10-xy_squre)*(y_2-y_SN10-xy_squre)))*Delta_xy+1
            d_k2_SN11 = (math.sqrt((x_2+xy_squre-x_SN11)*(x_2+xy_squre-x_SN11)+(y_2-y_SN11-xy_squre)*(y_2-y_SN11-xy_squre)))*Delta_xy+1
            d_k2_SN12 = (math.sqrt((x_2+xy_squre-x_SN12)*(x_2+xy_squre-x_SN12)+(y_2-y_SN12-xy_squre)*(y_2-y_SN12-xy_squre)))*Delta_xy+1
            beta_k2_SN10 = math.pow(10, -3)*math.pow(d_k2_SN10, -2.2)
            beta_k2_SN11 = math.pow(10, -3)*math.pow(d_k2_SN11, -2.2)
            beta_k2_SN12 = math.pow(10, -3)*math.pow(d_k2_SN12, -2.2) 
            theta_k2_SN10 = math.exp(1)**(-1*1j*(math.pi*(x_SN10-xy_squre-x_2)/d_k2_SN10))
            theta_k2_SN11 = math.exp(1)**(-1*1j*(math.pi*(x_SN11-xy_squre-x_2)/d_k2_SN11))
            theta_k2_SN12 = math.exp(1)**(-1*1j*(math.pi*(x_SN12-xy_squre-x_2)/d_k2_SN12))
            g_k2_SN10_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN10), math.exp(1)**(-1*1j*theta_k2_SN10*2), math.exp(1)**(-1*1j*theta_k2_SN10*3)])
            g_k2_SN11_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN11), math.exp(1)**(-1*1j*theta_k2_SN11*2), math.exp(1)**(-1*1j*theta_k2_SN11*3)])
            g_k2_SN12_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN12), math.exp(1)**(-1*1j*theta_k2_SN12*2), math.exp(1)**(-1*1j*theta_k2_SN12*3)])
            g_k2_SN10_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN11_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN12_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k2_SN10 = g_k2_SN10_LOS + g_k2_SN10_NLOS
            g_k2_SN11 = g_k2_SN11_LOS + g_k2_SN11_NLOS
            g_k2_SN12 = g_k2_SN12_LOS + g_k2_SN12_NLOS
            h_k2_SN10 = math.sqrt(beta_k2_SN10)*g_k2_SN10            
            h_k2_SN11 = math.sqrt(beta_k2_SN11)*g_k2_SN11
            h_k2_SN12 = math.sqrt(beta_k2_SN12)*g_k2_SN12
            H_k2_PART4 = np.array([h_k2_SN10[0],h_k2_SN10[1],h_k2_SN10[2],h_k2_SN10[3],h_k2_SN11[0],h_k2_SN11[1],h_k2_SN11[2],h_k2_SN11[3],h_k2_SN12[0],h_k2_SN12[1],h_k2_SN12[2],h_k2_SN12[3]]).T

            
            H_k2_PART3 = np.array([h_k2_SN7[0],h_k2_SN7[1],h_k2_SN7[2],h_k2_SN7[3],h_k2_SN8[0],h_k2_SN8[1],h_k2_SN8[2],h_k2_SN8[3],h_k2_SN9[0],h_k2_SN9[1],h_k2_SN9[2],h_k2_SN9[3]]).T
            #communication model
            H_k2_PART1 = np.array([h_k2_SN1[0],h_k2_SN1[1],h_k2_SN1[2],h_k2_SN1[3],h_k2_SN2[0],h_k2_SN2[1],h_k2_SN2[2],h_k2_SN2[3],h_k2_SN3[0],h_k2_SN3[1],h_k2_SN3[2],h_k2_SN3[3]]).T
            H_k2_PART2 = np.array([h_k2_SN4[0],h_k2_SN4[1],h_k2_SN4[2],h_k2_SN4[3],h_k2_SN5[0],h_k2_SN5[1],h_k2_SN5[2],h_k2_SN5[3],h_k2_SN6[0],h_k2_SN6[1],h_k2_SN6[2],h_k2_SN6[3]]).T
            
            #communication model
            H_k2 = np.array([H_k2_PART1, H_k2_PART2, H_k2_PART3, H_k2_PART4]).T
            # try:
            #     W_bar_k2 = H_k2.dot(lg.inv(H_k2.conj().T.dot(H_k2)))
            # except:
            #     W_bar_k2 = W_bar_k2
            # else:
            #     W_bar_k2 = H_k2.dot(lg.inv(H_k2.conj().T.dot(H_k2)))
            W_bar_k2 = H_k2.dot(lg.inv(H_k2.conj().T.dot(H_k2)))
            # w_k2_k1 = W_bar_k2[:, 0]/(np.linalg.norm(W_bar_k2[:, 0]))
            w_k2_k2 = W_bar_k2[:, 1]/(np.linalg.norm(W_bar_k2[:, 1]))
            # w_k2_SN6 = W_bar_k2[:, 2]/(np.linalg.norm(W_bar_k2[:, 2]))
            # w_k2_SN1 = W_bar_k2[:, 3]/(np.linalg.norm(W_bar_k2[:, 3]))
            # w_k2_SN2 = W_bar_k2[:, 4]/(np.linalg.norm(W_bar_k2[:, 4]))
            # w_k2_SN3 = W_bar_k2[:, 5]/(np.linalg.norm(W_bar_k2[:, 5]))
            # w_check1 = np.linalg.norm(w_k1_SN4)
            # w_check2 = np.linalg.norm(w_k1_SN5)
            # w_check3 = np.linalg.norm(w_k1_SN6)
            w_k2_SN6=w_k2_k2[8:12]
            w_k2_SN5=w_k2_k2[4:8]
            w_k2_SN4=w_k2_k2[0:4]
            # w_k2_SN3=w_k2_k1[4:6]
            # w_k2_SN2=w_k2_k1[2:4]
            # w_k2_SN1=w_k2_k1[0:2]
            # w_check1 = np.linalg.norm(w_k1_SN4)
            # w_check2 = np.linalg.norm(w_k1_SN5)
            # w_check3 = np.linalg.norm(w_k1_SN6)
            order_k2_SN4 = w_k2_SN4.conj().T.dot(h_k2_SN4)
            order_k2_SN5 = w_k2_SN5.conj().T.dot(h_k2_SN5)
            order_k2_SN6 = w_k2_SN6.conj().T.dot(h_k2_SN6)
            order_k2_SN4_positive = np.linalg.norm(order_k2_SN4)
            order_k2_SN5_positive = np.linalg.norm(order_k2_SN5)
            order_k2_SN6_positive = np.linalg.norm(order_k2_SN6)
            # print(order_k1_SN1_positive, order_k1_SN5_positive, order_k1_SN3_positive)
    
    
            x_7 = observation_k7%xy_squre
            y_7 = int(observation_k7/xy_squre)
            d_k7_SN4 = (math.sqrt((x_7-x_SN4)*(x_7-x_SN4)+(y_7+xy_squre-y_SN4)*(y_7+xy_squre-y_SN4)))*Delta_xy+1
            d_k7_SN5 = (math.sqrt((x_7-x_SN5)*(x_7-x_SN5)+(y_7+xy_squre-y_SN5)*(y_7+xy_squre-y_SN5)))*Delta_xy+1
            d_k7_SN6 = (math.sqrt((x_7-x_SN6)*(x_7-x_SN6)+(y_7+xy_squre-y_SN6)*(y_7+xy_squre-y_SN6)))*Delta_xy+1
            d_k7_SN1 = (math.sqrt((x_7+xy_squre-x_SN1)*(x_7+xy_squre-x_SN1)+(y_7+xy_squre-y_SN1)*(y_7+xy_squre-y_SN1)))*Delta_xy+1
            d_k7_SN2 = (math.sqrt((x_7+xy_squre-x_SN2)*(x_7+xy_squre-x_SN2)+(y_7+xy_squre-y_SN2)*(y_7+xy_squre-y_SN2)))*Delta_xy+1
            d_k7_SN3 = (math.sqrt((x_7+xy_squre-x_SN3)*(x_7+xy_squre-x_SN3)+(y_7+xy_squre-y_SN3)*(y_7+xy_squre-y_SN3)))*Delta_xy+1
            d_k7_destination = (math.sqrt((x_7-x_k7_destination)*(x_7-x_k7_destination)+(y_7-y_k7_destination)*(y_7-y_k7_destination)))*Delta_xy
            beta_k7_SN4 = math.pow(10, -3)*math.pow(d_k7_SN4, -2.2)
            beta_k7_SN5 = math.pow(10, -3)*math.pow(d_k7_SN5, -2.2)
            beta_k7_SN6 = math.pow(10, -3)*math.pow(d_k7_SN6, -2.2)
            beta_k7_SN1 = math.pow(10, -3)*math.pow(d_k7_SN1, -2.2)
            beta_k7_SN2 = math.pow(10, -3)*math.pow(d_k7_SN2, -2.2)
            beta_k7_SN3 = math.pow(10, -3)*math.pow(d_k7_SN3, -2.2)
            theta_k7_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_7)/d_k7_SN4))
            theta_k7_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_7)/d_k7_SN5))
            theta_k7_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_7)/d_k7_SN6))
            theta_k7_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_7-xy_squre)/d_k7_SN1))
            theta_k7_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_7-xy_squre)/d_k7_SN2))
            theta_k7_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_7-xy_squre)/d_k7_SN3))
            g_k7_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN4), math.exp(1)**(-1*1j*theta_k7_SN4*2), math.exp(1)**(-1*1j*theta_k7_SN4*3)])
            g_k7_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN5), math.exp(1)**(-1*1j*theta_k7_SN5*2), math.exp(1)**(-1*1j*theta_k7_SN5*3)])
            g_k7_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN6), math.exp(1)**(-1*1j*theta_k7_SN6*2), math.exp(1)**(-1*1j*theta_k7_SN6*3)])
            g_k7_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN1), math.exp(1)**(-1*1j*theta_k7_SN1*2), math.exp(1)**(-1*1j*theta_k7_SN1*3)])
            g_k7_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN2), math.exp(1)**(-1*1j*theta_k7_SN2*2), math.exp(1)**(-1*1j*theta_k7_SN2*3)])
            g_k7_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN3), math.exp(1)**(-1*1j*theta_k7_SN3*2), math.exp(1)**(-1*1j*theta_k7_SN3*3)])
            g_k7_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN4 = g_k7_SN4_LOS + g_k7_SN4_NLOS
            g_k7_SN5 = g_k7_SN5_LOS + g_k7_SN5_NLOS
            g_k7_SN6 = g_k7_SN6_LOS + g_k7_SN6_NLOS
            g_k7_SN1 = g_k7_SN1_LOS + g_k7_SN1_NLOS
            g_k7_SN2 = g_k7_SN2_LOS + g_k7_SN2_NLOS
            g_k7_SN3 = g_k7_SN3_LOS + g_k7_SN3_NLOS
            h_k7_SN4 = math.sqrt(beta_k7_SN4)*g_k7_SN4            
            h_k7_SN5 = math.sqrt(beta_k7_SN5)*g_k7_SN5
            h_k7_SN6 = math.sqrt(beta_k7_SN6)*g_k7_SN6
            h_k7_SN1 = math.sqrt(beta_k7_SN1)*g_k7_SN1            
            h_k7_SN2 = math.sqrt(beta_k7_SN2)*g_k7_SN2
            h_k7_SN3 = math.sqrt(beta_k7_SN3)*g_k7_SN3
            
            d_k7_SN7 = (math.sqrt((x_7-x_SN7)*(x_7-x_SN7)+(y_7-y_SN7)*(y_7-y_SN7)))*Delta_xy+1
            d_k7_SN8 = (math.sqrt((x_7-x_SN8)*(x_7-x_SN8)+(y_7-y_SN8)*(y_7-y_SN8)))*Delta_xy+1
            d_k7_SN9 = (math.sqrt((x_7-x_SN9)*(x_7-x_SN9)+(y_7-y_SN9)*(y_7-y_SN9)))*Delta_xy+1
            beta_k7_SN7 = math.pow(10, -3)*math.pow(d_k7_SN7, -2.2)
            beta_k7_SN8 = math.pow(10, -3)*math.pow(d_k7_SN8, -2.2)
            beta_k7_SN9 = math.pow(10, -3)*math.pow(d_k7_SN9, -2.2) 
            theta_k7_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7-x_7)/d_k7_SN7))
            theta_k7_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8-x_7)/d_k7_SN8))
            theta_k7_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9-x_7)/d_k7_SN9))
            g_k7_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN7), math.exp(1)**(-1*1j*theta_k7_SN7*2), math.exp(1)**(-1*1j*theta_k7_SN7*3)])
            g_k7_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN8), math.exp(1)**(-1*1j*theta_k7_SN8*2), math.exp(1)**(-1*1j*theta_k7_SN8*3)])
            g_k7_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN9), math.exp(1)**(-1*1j*theta_k7_SN9*2), math.exp(1)**(-1*1j*theta_k7_SN9*3)])
            g_k7_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN7 = g_k7_SN7_LOS + g_k7_SN7_NLOS
            g_k7_SN8 = g_k7_SN8_LOS + g_k7_SN8_NLOS
            g_k7_SN9 = g_k7_SN9_LOS + g_k7_SN9_NLOS
            h_k7_SN7 = math.sqrt(beta_k7_SN7)*g_k7_SN7            
            h_k7_SN8 = math.sqrt(beta_k7_SN8)*g_k7_SN8
            h_k7_SN9 = math.sqrt(beta_k7_SN9)*g_k7_SN9
            
            d_k7_SN10 = (math.sqrt((x_7+xy_squre-x_SN10)*(x_7+xy_squre-x_SN10)+(y_7-y_SN10)*(y_7-y_SN10)))*Delta_xy+1
            d_k7_SN11 = (math.sqrt((x_7+xy_squre-x_SN11)*(x_7+xy_squre-x_SN11)+(y_7-y_SN11)*(y_7-y_SN11)))*Delta_xy+1
            d_k7_SN12 = (math.sqrt((x_7+xy_squre-x_SN12)*(x_7+xy_squre-x_SN12)+(y_7-y_SN12)*(y_7-y_SN12)))*Delta_xy+1
            beta_k7_SN10 = math.pow(10, -3)*math.pow(d_k7_SN10, -2.2)
            beta_k7_SN11 = math.pow(10, -3)*math.pow(d_k7_SN11, -2.2)
            beta_k7_SN12 = math.pow(10, -3)*math.pow(d_k7_SN12, -2.2) 
            theta_k7_SN10 = math.exp(1)**(-1*1j*(math.pi*(x_SN10-x_7-xy_squre)/d_k7_SN10))
            theta_k7_SN11 = math.exp(1)**(-1*1j*(math.pi*(x_SN11-x_7-xy_squre)/d_k7_SN11))
            theta_k7_SN12 = math.exp(1)**(-1*1j*(math.pi*(x_SN12-x_7-xy_squre)/d_k7_SN12))
            g_k7_SN10_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN10), math.exp(1)**(-1*1j*theta_k7_SN10*2), math.exp(1)**(-1*1j*theta_k7_SN10*2)])
            g_k7_SN11_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN11), math.exp(1)**(-1*1j*theta_k7_SN11*2), math.exp(1)**(-1*1j*theta_k7_SN11*2)])
            g_k7_SN12_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN12), math.exp(1)**(-1*1j*theta_k7_SN12*2), math.exp(1)**(-1*1j*theta_k7_SN12*2)])
            g_k7_SN10_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN11_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN12_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k7_SN10 = g_k7_SN10_LOS + g_k7_SN10_NLOS
            g_k7_SN11 = g_k7_SN11_LOS + g_k7_SN11_NLOS
            g_k7_SN12 = g_k7_SN12_LOS + g_k7_SN12_NLOS
            h_k7_SN10 = math.sqrt(beta_k7_SN10)*g_k7_SN10            
            h_k7_SN11 = math.sqrt(beta_k7_SN11)*g_k7_SN11
            h_k7_SN12 = math.sqrt(beta_k7_SN12)*g_k7_SN12
            H_k7_PART4 = np.array([h_k7_SN10[0],h_k7_SN10[1],h_k7_SN10[2],h_k7_SN10[3],h_k7_SN11[0],h_k7_SN11[1],h_k7_SN11[2],h_k7_SN11[3],h_k7_SN12[0],h_k7_SN12[1],h_k7_SN12[2],h_k7_SN12[3]]).T
 
            
            H_k7_PART3 = np.array([h_k7_SN7[0],h_k7_SN7[1],h_k7_SN7[2],h_k7_SN7[3],h_k7_SN8[0],h_k7_SN8[1],h_k7_SN8[2],h_k7_SN8[3],h_k7_SN9[0],h_k7_SN9[1],h_k7_SN9[2],h_k7_SN9[3]]).T
            #communication model
            H_k7_PART1 = np.array([h_k7_SN1[0],h_k7_SN1[1],h_k7_SN1[2],h_k7_SN1[3],h_k7_SN2[0],h_k7_SN2[1],h_k7_SN2[2],h_k7_SN2[3],h_k7_SN3[0],h_k7_SN3[1],h_k7_SN3[2],h_k7_SN3[3]]).T
            H_k7_PART2 = np.array([h_k7_SN4[0],h_k7_SN4[1],h_k7_SN4[2],h_k7_SN4[3],h_k7_SN5[0],h_k7_SN5[1],h_k7_SN5[2],h_k7_SN5[3],h_k7_SN6[0],h_k7_SN6[1],h_k7_SN6[2],h_k7_SN6[3]]).T
            
            #communication model
            H_k7 = np.array([H_k7_PART1, H_k7_PART2, H_k7_PART3, H_k7_PART4]).T
            # try:
            #     W_bar_k7 = H_k7.dot(lg.inv(H_k7.conj().T.dot(H_k7)))
            # except:
            #     W_bar_k7 = W_bar_k7
            # else:
            #     W_bar_k7 = H_k7.dot(lg.inv(H_k7.conj().T.dot(H_k7)))
            W_bar_k7 = H_k7.dot(lg.inv(H_k7.conj().T.dot(H_k7)))
            # w_k7_k1 = W_bar_k7[:, 0]/(np.linalg.norm(W_bar_k7[:, 0]))
            w_k7_k7 = W_bar_k7[:, 2]/(np.linalg.norm(W_bar_k7[:, 2]))
            # w_k7_SN6 = W_bar_k7[:, 2]/(np.linalg.norm(W_bar_k7[:, 2]))
            # w_k7_SN1 = W_bar_k7[:, 3]/(np.linalg.norm(W_bar_k7[:, 3]))
            # w_k7_SN2 = W_bar_k7[:, 4]/(np.linalg.norm(W_bar_k7[:, 4]))
            # w_k7_SN3 = W_bar_k7[:, 5]/(np.linalg.norm(W_bar_k7[:, 5]))
            # w_check1 = np.linalg.norm(w_k1_SN4)
            # w_check7 = np.linalg.norm(w_k1_SN5)
            # w_check3 = np.linalg.norm(w_k1_SN6)
            w_k7_SN9=w_k7_k7[8:12]
            w_k7_SN8=w_k7_k7[4:8]
            w_k7_SN7=w_k7_k7[0:4]
            # w_k7_SN3=w_k7_k1[4:6]
            # w_k7_SN2=w_k7_k1[2:4]
            # w_k7_SN1=w_k7_k1[0:2]
            # w_check1 = np.linalg.norm(w_k1_SN4)
            # w_check7 = np.linalg.norm(w_k1_SN5)
            # w_check3 = np.linalg.norm(w_k1_SN6)
            order_k7_SN7 = w_k7_SN7.conj().T.dot(h_k7_SN7)
            order_k7_SN8 = w_k7_SN8.conj().T.dot(h_k7_SN8)
            order_k7_SN9 = w_k7_SN9.conj().T.dot(h_k7_SN9)
            order_k7_SN7_positive = np.linalg.norm(order_k7_SN7)
            order_k7_SN8_positive = np.linalg.norm(order_k7_SN8)
            order_k7_SN9_positive = np.linalg.norm(order_k7_SN9)
    
    
            x_10 = observation_k10%xy_squre
            y_10 = int(observation_k10/xy_squre)
            d_k10_SN4 = (math.sqrt((x_10-xy_squre-x_SN4)*(x_10-xy_squre-x_SN4)+(y_10+xy_squre-y_SN4)*(y_10+xy_squre-y_SN4)))*Delta_xy+1
            d_k10_SN5 = (math.sqrt((x_10-xy_squre-x_SN5)*(x_10-xy_squre-x_SN5)+(y_10+xy_squre-y_SN5)*(y_10+xy_squre-y_SN5)))*Delta_xy+1
            d_k10_SN6 = (math.sqrt((x_10-xy_squre-x_SN6)*(x_10-xy_squre-x_SN6)+(y_10+xy_squre-y_SN6)*(y_10+xy_squre-y_SN6)))*Delta_xy+1
            d_k10_SN1 = (math.sqrt((x_10-x_SN1)*(x_10-x_SN1)+(y_10+xy_squre-y_SN1)*(y_10+xy_squre-y_SN1)))*Delta_xy+1
            d_k10_SN2 = (math.sqrt((x_10-x_SN2)*(x_10-x_SN2)+(y_10+xy_squre-y_SN2)*(y_10+xy_squre-y_SN2)))*Delta_xy+1
            d_k10_SN3 = (math.sqrt((x_10-x_SN3)*(x_10-x_SN3)+(y_10+xy_squre-y_SN3)*(y_10+xy_squre-y_SN3)))*Delta_xy+1
            d_k10_destination = (math.sqrt((x_10-x_k10_destination)*(x_10-x_k10_destination)+(y_10-y_k10_destination)*(y_10-y_k10_destination)))*Delta_xy
            beta_k10_SN4 = math.pow(10, -3)*math.pow(d_k10_SN4, -2.2)
            beta_k10_SN5 = math.pow(10, -3)*math.pow(d_k10_SN5, -2.2)
            beta_k10_SN6 = math.pow(10, -3)*math.pow(d_k10_SN6, -2.2)
            beta_k10_SN1 = math.pow(10, -3)*math.pow(d_k10_SN1, -2.2)
            beta_k10_SN2 = math.pow(10, -3)*math.pow(d_k10_SN2, -2.2)
            beta_k10_SN3 = math.pow(10, -3)*math.pow(d_k10_SN3, -2.2)
            theta_k10_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_10+xy_squre)/d_k10_SN4))
            theta_k10_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_10+xy_squre)/d_k10_SN5))
            theta_k10_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_10+xy_squre)/d_k10_SN6))
            theta_k10_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_10)/d_k10_SN1))
            theta_k10_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_10)/d_k10_SN2))
            theta_k10_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_10)/d_k10_SN3))
            g_k10_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN4), math.exp(1)**(-1*1j*theta_k10_SN4*2), math.exp(1)**(-1*1j*theta_k10_SN4*3)])
            g_k10_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN5), math.exp(1)**(-1*1j*theta_k10_SN5*2), math.exp(1)**(-1*1j*theta_k10_SN5*3)])
            g_k10_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN6), math.exp(1)**(-1*1j*theta_k10_SN6*2), math.exp(1)**(-1*1j*theta_k10_SN6*3)])
            g_k10_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN1), math.exp(1)**(-1*1j*theta_k10_SN1*2), math.exp(1)**(-1*1j*theta_k10_SN1*3)])
            g_k10_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN2), math.exp(1)**(-1*1j*theta_k10_SN2*2), math.exp(1)**(-1*1j*theta_k10_SN2*3)])
            g_k10_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN3), math.exp(1)**(-1*1j*theta_k10_SN3*2), math.exp(1)**(-1*1j*theta_k10_SN3*3)])
            g_k10_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN4 = g_k10_SN4_LOS + g_k10_SN4_NLOS
            g_k10_SN5 = g_k10_SN5_LOS + g_k10_SN5_NLOS
            g_k10_SN6 = g_k10_SN6_LOS + g_k10_SN6_NLOS
            g_k10_SN1 = g_k10_SN1_LOS + g_k10_SN1_NLOS
            g_k10_SN2 = g_k10_SN2_LOS + g_k10_SN2_NLOS
            g_k10_SN3 = g_k10_SN3_LOS + g_k10_SN3_NLOS
            h_k10_SN4 = math.sqrt(beta_k10_SN4)*g_k10_SN4            
            h_k10_SN5 = math.sqrt(beta_k10_SN5)*g_k10_SN5
            h_k10_SN6 = math.sqrt(beta_k10_SN6)*g_k10_SN6
            h_k10_SN1 = math.sqrt(beta_k10_SN1)*g_k10_SN1            
            h_k10_SN2 = math.sqrt(beta_k10_SN2)*g_k10_SN2
            h_k10_SN3 = math.sqrt(beta_k10_SN3)*g_k10_SN3
            
            d_k10_SN7 = (math.sqrt((x_10-xy_squre-x_SN7)*(x_10-xy_squre-x_SN7)+(y_10-y_SN7)*(y_10-y_SN7)))*Delta_xy+1
            d_k10_SN8 = (math.sqrt((x_10-xy_squre-x_SN8)*(x_10-xy_squre-x_SN8)+(y_10-y_SN8)*(y_10-y_SN8)))*Delta_xy+1
            d_k10_SN9 = (math.sqrt((x_10-xy_squre-x_SN9)*(x_10-xy_squre-x_SN9)+(y_10-y_SN9)*(y_10-y_SN9)))*Delta_xy+1
            beta_k10_SN7 = math.pow(10, -3)*math.pow(d_k10_SN7, -2.2)
            beta_k10_SN8 = math.pow(10, -3)*math.pow(d_k10_SN8, -2.2)
            beta_k10_SN9 = math.pow(10, -3)*math.pow(d_k10_SN9, -2.2) 
            theta_k10_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7-x_10+xy_squre)/d_k10_SN7))
            theta_k10_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8-x_10+xy_squre)/d_k10_SN8))
            theta_k10_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9-x_10+xy_squre)/d_k10_SN9))
            g_k10_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN7), math.exp(1)**(-1*1j*theta_k10_SN7*2), math.exp(1)**(-1*1j*theta_k10_SN7*3)])
            g_k10_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN8), math.exp(1)**(-1*1j*theta_k10_SN8*2), math.exp(1)**(-1*1j*theta_k10_SN8*3)])
            g_k10_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN9), math.exp(1)**(-1*1j*theta_k10_SN9*2), math.exp(1)**(-1*1j*theta_k10_SN9*3)])
            g_k10_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN7 = g_k10_SN7_LOS + g_k10_SN7_NLOS
            g_k10_SN8 = g_k10_SN8_LOS + g_k10_SN8_NLOS
            g_k10_SN9 = g_k10_SN9_LOS + g_k10_SN9_NLOS
            h_k10_SN7 = math.sqrt(beta_k10_SN7)*g_k10_SN7            
            h_k10_SN8 = math.sqrt(beta_k10_SN8)*g_k10_SN8
            h_k10_SN9 = math.sqrt(beta_k10_SN9)*g_k10_SN9
            
            d_k10_SN10 = (math.sqrt((x_10-x_SN10)*(x_10-x_SN10)+(y_10-y_SN10)*(y_10-y_SN10)))*Delta_xy+1
            d_k10_SN11 = (math.sqrt((x_10-x_SN11)*(x_10-x_SN11)+(y_10-y_SN11)*(y_10-y_SN11)))*Delta_xy+1
            d_k10_SN12 = (math.sqrt((x_10-x_SN12)*(x_10-x_SN12)+(y_10-y_SN12)*(y_10-y_SN12)))*Delta_xy+1
            beta_k10_SN10 = math.pow(10, -3)*math.pow(d_k10_SN10, -2.2)
            beta_k10_SN11 = math.pow(10, -3)*math.pow(d_k10_SN11, -2.2)
            beta_k10_SN12 = math.pow(10, -3)*math.pow(d_k10_SN12, -2.2) 
            theta_k10_SN10 = math.exp(1)**(-1*1j*(math.pi*(x_SN10-x_10)/d_k10_SN10))
            theta_k10_SN11 = math.exp(1)**(-1*1j*(math.pi*(x_SN11-x_10)/d_k10_SN11))
            theta_k10_SN12 = math.exp(1)**(-1*1j*(math.pi*(x_SN12-x_10)/d_k10_SN12))
            g_k10_SN10_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN10), math.exp(1)**(-1*1j*theta_k10_SN10*2), math.exp(1)**(-1*1j*theta_k10_SN10*2)])
            g_k10_SN11_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN11), math.exp(1)**(-1*1j*theta_k10_SN11*2), math.exp(1)**(-1*1j*theta_k10_SN11*2)])
            g_k10_SN12_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN12), math.exp(1)**(-1*1j*theta_k10_SN12*2), math.exp(1)**(-1*1j*theta_k10_SN12*2)])
            g_k10_SN10_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN11_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN12_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
            g_k10_SN10 = g_k10_SN10_LOS + g_k10_SN10_NLOS
            g_k10_SN11 = g_k10_SN11_LOS + g_k10_SN11_NLOS
            g_k10_SN12 = g_k10_SN12_LOS + g_k10_SN12_NLOS
            h_k10_SN10 = math.sqrt(beta_k10_SN10)*g_k10_SN10            
            h_k10_SN11 = math.sqrt(beta_k10_SN11)*g_k10_SN11
            h_k10_SN12 = math.sqrt(beta_k10_SN12)*g_k10_SN12
            H_k10_PART4 = np.array([h_k10_SN10[0],h_k10_SN10[1],h_k10_SN10[2],h_k10_SN10[3],h_k10_SN11[0],h_k10_SN11[1],h_k10_SN11[2],h_k10_SN11[3],h_k10_SN12[0],h_k10_SN12[1],h_k10_SN12[2],h_k10_SN12[3]]).T
 
            
            H_k10_PART3 = np.array([h_k10_SN7[0],h_k10_SN7[1],h_k10_SN7[2],h_k10_SN7[3],h_k10_SN8[0],h_k10_SN8[1],h_k10_SN8[2],h_k10_SN8[3],h_k10_SN9[0],h_k10_SN9[1],h_k10_SN9[2],h_k10_SN9[3]]).T
            #communication model
            H_k10_PART1 = np.array([h_k10_SN1[0],h_k10_SN1[1],h_k10_SN1[2],h_k10_SN1[3],h_k10_SN2[0],h_k10_SN2[1],h_k10_SN2[2],h_k10_SN2[3],h_k10_SN3[0],h_k10_SN3[1],h_k10_SN3[2],h_k10_SN3[3]]).T
            H_k10_PART2 = np.array([h_k10_SN4[0],h_k10_SN4[1],h_k10_SN4[2],h_k10_SN4[3],h_k10_SN5[0],h_k10_SN5[1],h_k10_SN5[2],h_k10_SN5[3],h_k10_SN6[0],h_k10_SN6[1],h_k10_SN6[2],h_k10_SN6[3]]).T
            
            #communication model
            H_k10 = np.array([H_k10_PART1, H_k10_PART2, H_k10_PART3, H_k10_PART4]).T
            # try:
            #     W_bar_k7 = H_k7.dot(lg.inv(H_k7.conj().T.dot(H_k7)))
            # except:
            #     W_bar_k7 = W_bar_k7
            # else:
            #     W_bar_k7 = H_k7.dot(lg.inv(H_k7.conj().T.dot(H_k7)))
            W_bar_k10 = H_k10.dot(lg.inv(H_k10.conj().T.dot(H_k10)))
            # w_k7_k1 = W_bar_k7[:, 0]/(np.linalg.norm(W_bar_k7[:, 0]))
            w_k10_k10 = W_bar_k10[:, 3]/(np.linalg.norm(W_bar_k10[:, 3]))
            # w_k7_SN6 = W_bar_k7[:, 2]/(np.linalg.norm(W_bar_k7[:, 2]))
            # w_k7_SN1 = W_bar_k7[:, 3]/(np.linalg.norm(W_bar_k7[:, 3]))
            # w_k7_SN2 = W_bar_k7[:, 4]/(np.linalg.norm(W_bar_k7[:, 4]))
            # w_k7_SN3 = W_bar_k7[:, 5]/(np.linalg.norm(W_bar_k7[:, 5]))
            # w_check1 = np.linalg.norm(w_k1_SN4)
            # w_check7 = np.linalg.norm(w_k1_SN5)
            # w_check3 = np.linalg.norm(w_k1_SN6)
            w_k10_SN12=w_k10_k10[8:12]
            w_k10_SN11=w_k10_k10[4:8]
            w_k10_SN10=w_k10_k10[0:4]
            # w_k7_SN3=w_k7_k1[4:6]
            # w_k7_SN2=w_k7_k1[2:4]
            # w_k7_SN1=w_k7_k1[0:2]
            # w_check1 = np.linalg.norm(w_k1_SN4)
            # w_check7 = np.linalg.norm(w_k1_SN5)
            # w_check3 = np.linalg.norm(w_k1_SN6)
            order_k10_SN10 = w_k10_SN10.conj().T.dot(h_k10_SN10)
            order_k10_SN11 = w_k10_SN11.conj().T.dot(h_k10_SN11)
            order_k10_SN12 = w_k10_SN12.conj().T.dot(h_k10_SN12)
            order_k10_SN10_positive = np.linalg.norm(order_k10_SN10)
            order_k10_SN11_positive = np.linalg.norm(order_k10_SN11)
            order_k10_SN12_positive = np.linalg.norm(order_k10_SN12)
    
    
    
            # observation_k12 = observation_k1 + observation_k7*xy_squre*xy_squre
            # RL choose action based on observation_k1
            if k1_done_label == 0:
                action_k1 = RL1.choose_action(observation_k1)
            if k2_done_label == 0:
                action_k2 = RL2.choose_action(observation_k2)
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
SINR_k1_SN{} = (order_k1_SN{}_positive**2*p_k1_SN{})/((np.linalg.norm(w_k1_SN{}.conj().T.dot(h_k1_SN{})))**2*p_k1_SN{}+10**(-9))
SINR_k1_SN{} = (order_k1_SN{}_positive**2*p_k1_SN{})/((np.linalg.norm(w_k1_SN{}.conj().T.dot(h_k1_SN{})))**2*p_k1_SN{}+(np.linalg.norm(w_k1_SN{}.conj().T.dot(h_k1_SN{})))**2*p_k1_SN{}+10**(-9))'''.format(order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[1]+1,order_k1_index[1]+1,order_k1_index[1]+1,order_k1_index[1]+1,order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[0]+1,order_k1_index[0]+1,order_k1_index[0]+1,order_k1_index[0]+1,order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[0]+1,order_k1_index[1]+1,order_k1_index[1]+1))
    
            order_k2 = [order_k2_SN4_positive, order_k2_SN5_positive, order_k2_SN6_positive]
            order_k2_index = [order_k2[0] for order_k2 in sorted(enumerate(order_k2),key=lambda i:i[1], reverse=True)]
            p_k2_SN_index = M_k
            for i in order_k2_index:
                exec('''p_k2_SN{}_int = p_c63[int(action_k2/5), p_k2_SN_index-1]
p_k2_SN{} = p_k2_SN{}_int*P_SN_mav/(6+1)
p_k2_SN_index = p_k2_SN_index-1'''.format(i+4, i+4, i+4))
            
            order_k2_index = [order_k2[0] for order_k2 in sorted(enumerate(order_k2),key=lambda i:i[1], reverse=True)]
            exec('''SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/(10**(-9))
SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/((np.linalg.norm(w_k2_SN{}.conj().T.dot(h_k2_SN{})))**2*p_k2_SN{}+10**(-9))
SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/((np.linalg.norm(w_k2_SN{}.conj().T.dot(h_k2_SN{})))**2*p_k2_SN{}+(np.linalg.norm(w_k2_SN{}.conj().T.dot(h_k2_SN{})))**2*p_k2_SN{}+10**(-9))'''.format(order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[1]+4,order_k2_index[1]+4,order_k2_index[1]+4,order_k2_index[1]+4,order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[0]+4,order_k2_index[0]+4,order_k2_index[0]+4,order_k2_index[0]+4,order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[0]+4,order_k2_index[1]+4,order_k2_index[1]+4))
            
            # print(np.linalg.norm(w_k2_SN4.conj().T.dot(h_k2_SN2)), np.linalg.norm(w_k2_SN5.conj().T.dot(h_k2_SN4)), np.linalg.norm(w_k2_SN4.conj().T.dot(h_k2_SN1)), np.linalg.norm(w_k2_SN6.conj().T.dot(h_k2_SN4)))
            # print(np.linalg.norm(w_k1_SN1.conj().T.dot(h_k1_SN2)), np.linalg.norm(w_k1_SN1.conj().T.dot(h_k1_SN4)), np.linalg.norm(w_k1_SN2.conj().T.dot(h_k1_SN5)), np.linalg.norm(w_k1_SN2.conj().T.dot(h_k1_SN4)))

            if k7_done_label == 0:
                action_k7 = RL7.choose_action(observation_k7)
            # action_k1 = action_k12%ACTION_SHAPE_single
            # action_k2 = int(action_k12/ACTION_SHAPE_single)
            order_k7 = [order_k7_SN7_positive, order_k7_SN8_positive, order_k7_SN9_positive]
            order_k7_index = [order_k7[0] for order_k7 in sorted(enumerate(order_k7),key=lambda i:i[1], reverse=True)]
            p_k7_SN_index = M_k
            for i in order_k7_index:
                exec('''p_k7_SN{}_int = p_c63[int(action_k7/5), p_k7_SN_index-1]
p_k7_SN{} = p_k7_SN{}_int*P_SN_mav/(6+1)
p_k7_SN_index = p_k7_SN_index-1'''.format(i+7, i+7, i+7))
            
            order_k7_index = [order_k7[0] for order_k7 in sorted(enumerate(order_k7),key=lambda i:i[1], reverse=True)]
            exec('''SINR_k7_SN{} = (order_k7_SN{}_positive**2*p_k7_SN{})/(10**(-9))
SINR_k7_SN{} = (order_k7_SN{}_positive**2*p_k7_SN{})/((np.linalg.norm(w_k7_SN{}.conj().T.dot(h_k7_SN{})))**2*p_k7_SN{}+10**(-9))
SINR_k7_SN{} = (order_k7_SN{}_positive**2*p_k7_SN{})/((np.linalg.norm(w_k7_SN{}.conj().T.dot(h_k7_SN{})))**2*p_k7_SN{}+(np.linalg.norm(w_k7_SN{}.conj().T.dot(h_k7_SN{})))**2*p_k7_SN{}+10**(-9))'''.format(order_k7_index[2]+7,order_k7_index[2]+7,order_k7_index[2]+7,order_k7_index[1]+7,order_k7_index[1]+7,order_k7_index[1]+7,order_k7_index[1]+7,order_k7_index[2]+7,order_k7_index[2]+7,order_k7_index[0]+7,order_k7_index[0]+7,order_k7_index[0]+7,order_k7_index[0]+7,order_k7_index[2]+7,order_k7_index[2]+7,order_k7_index[0]+7,order_k7_index[1]+7,order_k7_index[1]+7))
            R_k7_SN7 = math.log((1+SINR_k7_SN7),2)
            R_k7_SN8 = math.log((1+SINR_k7_SN8),2)
            R_k7_SN9 = math.log((1+SINR_k7_SN9),2)
            R_k1_SN1 = math.log((1+SINR_k1_SN1),2)
            R_k1_SN2 = math.log((1+SINR_k1_SN2),2)
            R_k1_SN3 = math.log((1+SINR_k1_SN3),2)
            R_k2_SN4 = math.log((1+SINR_k2_SN4),2)
            R_k2_SN5 = math.log((1+SINR_k2_SN5),2)
            R_k2_SN6 = math.log((1+SINR_k2_SN6),2)
            observation_k7_, reward_k7, done7 = env7.step(action_k7)
            reward_k7 = (0.00005)*min((R_k7_SN7+R_k7_SN8+R_k7_SN9),50)
            
            
            if k10_done_label == 0:
                action_k10 = RL10.choose_action(observation_k10)
            # action_k1 = action_k12%ACTION_SHAPE_single
            # action_k2 = int(action_k12/ACTION_SHAPE_single)
            order_k10 = [order_k10_SN10_positive, order_k10_SN11_positive, order_k10_SN12_positive]
            order_k10_index = [order_k10[0] for order_k10 in sorted(enumerate(order_k10),key=lambda i:i[1], reverse=True)]
            p_k10_SN_index = M_k
            for i in order_k10_index:
                exec('''p_k10_SN{}_int = p_c63[int(action_k10/5), p_k10_SN_index-1]
p_k10_SN{} = p_k10_SN{}_int*P_SN_mav/(6+1)
p_k10_SN_index = p_k10_SN_index-1'''.format(i+10, i+10, i+10))
            
            order_k10_index = [order_k10[0] for order_k10 in sorted(enumerate(order_k10),key=lambda i:i[1], reverse=True)]
            exec('''SINR_k10_SN{} = (order_k10_SN{}_positive**2*p_k10_SN{})/(10**(-9))
SINR_k10_SN{} = (order_k10_SN{}_positive**2*p_k10_SN{})/((np.linalg.norm(w_k10_SN{}.conj().T.dot(h_k10_SN{})))**2*p_k10_SN{}+10**(-9))
SINR_k10_SN{} = (order_k10_SN{}_positive**2*p_k10_SN{})/((np.linalg.norm(w_k10_SN{}.conj().T.dot(h_k10_SN{})))**2*p_k10_SN{}+(np.linalg.norm(w_k10_SN{}.conj().T.dot(h_k10_SN{})))**2*p_k10_SN{}+10**(-9))'''.format(order_k10_index[2]+10,order_k10_index[2]+10,order_k10_index[2]+10,order_k10_index[1]+10,order_k10_index[1]+10,order_k10_index[1]+10,order_k10_index[1]+10,order_k10_index[2]+10,order_k10_index[2]+10,order_k10_index[0]+10,order_k10_index[0]+10,order_k10_index[0]+10,order_k10_index[0]+10,order_k10_index[2]+10,order_k10_index[2]+10,order_k10_index[0]+10,order_k10_index[1]+10,order_k10_index[1]+10))
            R_k10_SN10 = math.log((1+SINR_k10_SN10),2)
            R_k10_SN11 = math.log((1+SINR_k10_SN11),2)
            R_k10_SN12 = math.log((1+SINR_k10_SN12),2)
            observation_k10_, reward_k10, done10 = env10.step(action_k10)
            reward_k10 = (0.00005)*min((R_k10_SN10+R_k10_SN11+R_k10_SN12),50)
            ##########################video demo
            SINR_matrix_k7=np.zeros([240, 240])
            # SINR_matrix_k7[240,0]=0
            # SINR_matrix_k7[240,0]=0
            if episode>4950:
                for x0_index in range(120):
                    for y0_index in range(120):
                        x_index=x0_index/6
                        y_index=y0_index/6
                        d_k7_SN7 = (math.sqrt((x_index-x_SN7)*(x_index-x_SN7)+(y_index-y_SN7)*(y_index-y_SN7)))*0.5+1
                        d_k7_SN8 = (math.sqrt((x_index-x_SN8)*(x_index-x_SN8)+(y_index-y_SN8)*(y_index-y_SN8)))*0.5+1
                        d_k7_SN9 = (math.sqrt((x_index-x_SN9)*(x_index-x_SN9)+(y_index-y_SN9)*(y_index-y_SN9)))*0.5+1
                        beta_k7_SN7 = math.pow(10, -3)*math.pow(d_k7_SN7, -2.2)
                        beta_k7_SN8 = math.pow(10, -3)*math.pow(d_k7_SN8, -2.2)
                        beta_k7_SN9 = math.pow(10, -3)*math.pow(d_k7_SN9, -2.2) 
                        theta_k7_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_index)/d_k7_SN7))
                        theta_k7_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_index)/d_k7_SN8))
                        theta_k7_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_index)/d_k7_SN9))
                        g_k7_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN7), math.exp(1)**(-1*1j*theta_k7_SN7*2), math.exp(1)**(-1*1j*theta_k7_SN7*3)])
                        g_k7_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN8), math.exp(1)**(-1*1j*theta_k7_SN8*2), math.exp(1)**(-1*1j*theta_k7_SN8*3)])
                        g_k7_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k7_SN9), math.exp(1)**(-1*1j*theta_k7_SN9*2), math.exp(1)**(-1*1j*theta_k7_SN9*3)])
                        g_k7_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k7_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k7_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k7_SN7 = g_k7_SN7_LOS + g_k7_SN7_NLOS
                        g_k7_SN8 = g_k7_SN8_LOS + g_k7_SN8_NLOS
                        g_k7_SN9 = g_k7_SN9_LOS + g_k7_SN9_NLOS
                        h_k7_SN7 = math.sqrt(beta_k7_SN7)*g_k7_SN7            
                        h_k7_SN8 = math.sqrt(beta_k7_SN8)*g_k7_SN8
                        h_k7_SN9 = math.sqrt(beta_k7_SN9)*g_k7_SN9
                        order_k7_SN7 = w_k7_SN7.conj().T.dot(h_k7_SN7)
                        order_k7_SN8 = w_k7_SN8.conj().T.dot(h_k7_SN8)
                        order_k7_SN9 = w_k7_SN9.conj().T.dot(h_k7_SN9)
                        order_k7_SN7_positive = np.linalg.norm(order_k7_SN7)
                        order_k7_SN8_positive = np.linalg.norm(order_k7_SN8)
                        order_k7_SN9_positive = np.linalg.norm(order_k7_SN9)
                        exec('''SINR_k7_SN{} = (order_k7_SN{}_positive**2*p_k7_SN{})/(10**(-9))
SINR_k7_SN{} = (order_k7_SN{}_positive**2*p_k7_SN{})/((np.linalg.norm(w_k7_SN{}.conj().T.dot(h_k7_SN{})))**2*p_k7_SN{}+10**(-9))
SINR_k7_SN{} = (order_k7_SN{}_positive**2*p_k7_SN{})/((np.linalg.norm(w_k7_SN{}.conj().T.dot(h_k7_SN{})))**2*p_k7_SN{}+(np.linalg.norm(w_k7_SN{}.conj().T.dot(h_k7_SN{})))**2*p_k7_SN{}+10**(-9))'''.format(order_k7_index[2]+7,order_k7_index[2]+7,order_k7_index[2]+7,order_k7_index[1]+7,order_k7_index[1]+7,order_k7_index[1]+7,order_k7_index[1]+7,order_k7_index[2]+7,order_k7_index[2]+7,order_k7_index[0]+7,order_k7_index[0]+7,order_k7_index[0]+7,order_k7_index[0]+7,order_k7_index[2]+7,order_k7_index[2]+7,order_k7_index[0]+7,order_k7_index[1]+7,order_k7_index[1]+7))
                        R_k7_SN7 = math.log((1+SINR_k7_SN7),2)
                        R_k7_SN8 = math.log((1+SINR_k7_SN8),2)
                        R_k7_SN9 = math.log((1+SINR_k7_SN9),2)
                        x_index=int(x_index*6+0.5)+120
                        y_index=int(y_index*6+0.5)+120
                        SINR_matrix_k7[y_index, x_index]=(h_k7_SN7[0]+h_k7_SN8[0]+h_k7_SN9[0]).real
                        ####################################################################################################robot1
                        x_index=x0_index/6
                        y_index=y0_index/6
                        d_k1_SN1 = (math.sqrt((x_index-x_SN1)*(x_index-x_SN1)+(y_index-y_SN1)*(y_index-y_SN1)))*0.5+1
                        d_k1_SN2 = (math.sqrt((x_index-x_SN2)*(x_index-x_SN2)+(y_index-y_SN2)*(y_index-y_SN2)))*0.5+1
                        d_k1_SN3 = (math.sqrt((x_index-x_SN3)*(x_index-x_SN3)+(y_index-y_SN3)*(y_index-y_SN3)))*0.5+1
                        beta_k1_SN1 = math.pow(10, -3)*math.pow(d_k1_SN1, -2.2)
                        beta_k1_SN2 = math.pow(10, -3)*math.pow(d_k1_SN2, -2.2)
                        beta_k1_SN3 = math.pow(10, -3)*math.pow(d_k1_SN3, -2.2) 
                        theta_k1_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1+xy_squre-x_index)/d_k1_SN1))
                        theta_k1_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2+xy_squre-x_index)/d_k1_SN2))
                        theta_k1_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3+xy_squre-x_index)/d_k1_SN3))
                        g_k1_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN1), math.exp(1)**(-1*1j*theta_k1_SN1*2), math.exp(1)**(-1*1j*theta_k1_SN1*3)])
                        g_k1_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN2), math.exp(1)**(-1*1j*theta_k1_SN2*2), math.exp(1)**(-1*1j*theta_k1_SN2*3)])
                        g_k1_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN3), math.exp(1)**(-1*1j*theta_k1_SN3*2), math.exp(1)**(-1*1j*theta_k1_SN3*3)])
                        g_k1_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k1_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k1_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k1_SN1 = g_k1_SN1_LOS + g_k1_SN1_NLOS
                        g_k1_SN2 = g_k1_SN2_LOS + g_k1_SN2_NLOS
                        g_k1_SN3 = g_k1_SN3_LOS + g_k1_SN3_NLOS
                        h_k1_SN1 = math.sqrt(beta_k1_SN1)*g_k1_SN1            
                        h_k1_SN2 = math.sqrt(beta_k1_SN2)*g_k1_SN2
                        h_k1_SN3 = math.sqrt(beta_k1_SN3)*g_k1_SN3
                        order_k1_SN1 = w_k1_SN1.conj().T.dot(h_k1_SN1)
                        order_k1_SN2 = w_k1_SN2.conj().T.dot(h_k1_SN2)
                        order_k1_SN3 = w_k1_SN3.conj().T.dot(h_k1_SN3)
                        order_k1_SN1_positive = np.linalg.norm(order_k1_SN1)
                        order_k1_SN2_positive = np.linalg.norm(order_k1_SN2)
                        order_k1_SN3_positive = np.linalg.norm(order_k1_SN3)
                        exec('''SINR_k1_SN{} = (order_k1_SN{}_positive**2*p_k1_SN{})/(10**(-9))
SINR_k1_SN{} = (order_k1_SN{}_positive**2*p_k1_SN{})/((np.linalg.norm(w_k1_SN{}.conj().T.dot(h_k1_SN{})))**2*p_k1_SN{}+10**(-9))
SINR_k1_SN{} = (order_k1_SN{}_positive**2*p_k1_SN{})/((np.linalg.norm(w_k1_SN{}.conj().T.dot(h_k1_SN{})))**2*p_k1_SN{}+(np.linalg.norm(w_k1_SN{}.conj().T.dot(h_k1_SN{})))**2*p_k1_SN{}+10**(-9))'''.format(order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[1]+1,order_k1_index[1]+1,order_k1_index[1]+1,order_k1_index[1]+1,order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[0]+1,order_k1_index[0]+1,order_k1_index[0]+1,order_k1_index[0]+1,order_k1_index[2]+1,order_k1_index[2]+1,order_k1_index[0]+1,order_k1_index[1]+1,order_k1_index[1]+1))
                        
                        R_k1_SN1 = math.log((1+SINR_k1_SN1),2)
                        R_k1_SN2 = math.log((1+SINR_k1_SN2),2)
                        R_k1_SN3 = math.log((1+SINR_k1_SN3),2)
                        x_index=int(x_index*6+0.5)
                        y_index=int(y_index*6+0.5)
                        SINR_matrix_k7[y_index, x_index]=(h_k1_SN1[0]+h_k1_SN2[0]+h_k1_SN3[0]).real
                        #################################################################################################### robot2
                        
                        x_index=x0_index/6
                        y_index=y0_index/6
                        d_k2_SN4 = (math.sqrt((x_index-x_SN4)*(x_index-x_SN4)+(y_index-y_SN4)*(y_index-y_SN4)))*0.5+1
                        d_k2_SN5 = (math.sqrt((x_index-x_SN5)*(x_index-x_SN5)+(y_index-y_SN5)*(y_index-y_SN5)))*0.5+1
                        d_k2_SN6 = (math.sqrt((x_index-x_SN6)*(x_index-x_SN6)+(y_index-y_SN6)*(y_index-y_SN6)))*0.5+1
                        beta_k2_SN4 = math.pow(10, -3)*math.pow(d_k2_SN4, -2.2)
                        beta_k2_SN5 = math.pow(10, -3)*math.pow(d_k2_SN5, -2.2)
                        beta_k2_SN6 = math.pow(10, -3)*math.pow(d_k2_SN6, -2.2) 
                        theta_k2_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4+xy_squre-x_index)/d_k2_SN4))
                        theta_k2_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5+xy_squre-x_index)/d_k2_SN5))
                        theta_k2_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6+xy_squre-x_index)/d_k2_SN6))
                        g_k2_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN4), math.exp(1)**(-1*1j*theta_k2_SN4*2), math.exp(1)**(-1*1j*theta_k2_SN4*3)])
                        g_k2_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN5), math.exp(1)**(-1*1j*theta_k2_SN5*2), math.exp(1)**(-1*1j*theta_k2_SN5*3)])
                        g_k2_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN6), math.exp(1)**(-1*1j*theta_k2_SN6*2), math.exp(1)**(-1*1j*theta_k2_SN6*3)])
                        g_k2_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k2_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k2_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k2_SN4 = g_k2_SN4_LOS + g_k2_SN4_NLOS
                        g_k2_SN5 = g_k2_SN5_LOS + g_k2_SN5_NLOS
                        g_k2_SN6 = g_k2_SN6_LOS + g_k2_SN6_NLOS
                        h_k2_SN4 = math.sqrt(beta_k2_SN4)*g_k2_SN4            
                        h_k2_SN5 = math.sqrt(beta_k2_SN5)*g_k2_SN5
                        h_k2_SN6 = math.sqrt(beta_k2_SN6)*g_k2_SN6
                        order_k2_SN4 = w_k2_SN4.conj().T.dot(h_k2_SN4)
                        order_k2_SN5 = w_k2_SN5.conj().T.dot(h_k2_SN5)
                        order_k2_SN6 = w_k2_SN6.conj().T.dot(h_k2_SN6)
                        order_k2_SN4_positive = np.linalg.norm(order_k2_SN4)
                        order_k2_SN5_positive = np.linalg.norm(order_k2_SN5)
                        order_k2_SN6_positive = np.linalg.norm(order_k2_SN6)
                        exec('''SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/(10**(-9))
SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/((np.linalg.norm(w_k2_SN{}.conj().T.dot(h_k2_SN{})))**2*p_k2_SN{}+10**(-9))
SINR_k2_SN{} = (order_k2_SN{}_positive**2*p_k2_SN{})/((np.linalg.norm(w_k2_SN{}.conj().T.dot(h_k2_SN{})))**2*p_k2_SN{}+(np.linalg.norm(w_k2_SN{}.conj().T.dot(h_k2_SN{})))**2*p_k2_SN{}+10**(-9))'''.format(order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[1]+4,order_k2_index[1]+4,order_k2_index[1]+4,order_k2_index[1]+4,order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[0]+4,order_k2_index[0]+4,order_k2_index[0]+4,order_k2_index[0]+4,order_k2_index[2]+4,order_k2_index[2]+4,order_k2_index[0]+4,order_k2_index[1]+4,order_k2_index[1]+4))
                       
                        R_k2_SN4 = math.log((1+SINR_k2_SN4),2)
                        R_k2_SN5 = math.log((1+SINR_k2_SN5),2)
                        R_k2_SN6 = math.log((1+SINR_k2_SN6),2)
                        x_index=int(x_index*6+0.5)+120
                        y_index=int(y_index*6+0.5)
                        SINR_matrix_k7[y_index, x_index]=(h_k2_SN4[0]+h_k2_SN5[0]+h_k2_SN6[0]).real
                        
                        #################################################################################################### robot10
                        
                        x_index=x0_index/6
                        y_index=y0_index/6
                        d_k10_SN10 = (math.sqrt((x_index-x_SN10)*(x_index-x_SN10)+(y_index-y_SN10)*(y_index-y_SN10)))*0.5+1
                        d_k10_SN11 = (math.sqrt((x_index-x_SN11)*(x_index-x_SN11)+(y_index-y_SN11)*(y_index-y_SN11)))*0.5+1
                        d_k10_SN12 = (math.sqrt((x_index-x_SN12)*(x_index-x_SN12)+(y_index-y_SN12)*(y_index-y_SN12)))*0.5+1
                        beta_k10_SN10 = math.pow(10, -3)*math.pow(d_k10_SN10, -2.2)
                        beta_k10_SN11 = math.pow(10, -3)*math.pow(d_k10_SN11, -2.2)
                        beta_k10_SN12 = math.pow(10, -3)*math.pow(d_k10_SN12, -2.2) 
                        theta_k10_SN10 = math.exp(1)**(-1*1j*(math.pi*(x_SN10+xy_squre-x_index)/d_k10_SN10))
                        theta_k10_SN11 = math.exp(1)**(-1*1j*(math.pi*(x_SN11+xy_squre-x_index)/d_k10_SN11))
                        theta_k10_SN12 = math.exp(1)**(-1*1j*(math.pi*(x_SN12+xy_squre-x_index)/d_k10_SN12))
                        g_k10_SN10_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN10), math.exp(1)**(-1*1j*theta_k10_SN10*2), math.exp(1)**(-1*1j*theta_k10_SN10*3)])
                        g_k10_SN11_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN11), math.exp(1)**(-1*1j*theta_k10_SN11*2), math.exp(1)**(-1*1j*theta_k10_SN11*3)])
                        g_k10_SN12_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k10_SN12), math.exp(1)**(-1*1j*theta_k10_SN12*2), math.exp(1)**(-1*1j*theta_k10_SN12*3)])
                        g_k10_SN10_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k10_SN11_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k10_SN12_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
                        g_k10_SN10 = g_k10_SN10_LOS + g_k10_SN10_NLOS
                        g_k10_SN11 = g_k10_SN11_LOS + g_k10_SN11_NLOS
                        g_k10_SN12 = g_k10_SN12_LOS + g_k10_SN12_NLOS
                        h_k10_SN10 = math.sqrt(beta_k10_SN10)*g_k10_SN10            
                        h_k10_SN11 = math.sqrt(beta_k10_SN11)*g_k10_SN11
                        h_k10_SN12 = math.sqrt(beta_k10_SN12)*g_k10_SN12
                        order_k10_SN10 = w_k10_SN10.conj().T.dot(h_k10_SN10)
                        order_k10_SN11 = w_k10_SN11.conj().T.dot(h_k10_SN11)
                        order_k10_SN12 = w_k10_SN12.conj().T.dot(h_k10_SN12)
                        order_k10_SN10_positive = np.linalg.norm(order_k10_SN10)
                        order_k10_SN11_positive = np.linalg.norm(order_k10_SN11)
                        order_k10_SN12_positive = np.linalg.norm(order_k10_SN12)
                        exec('''SINR_k10_SN{} = (order_k10_SN{}_positive**2*p_k10_SN{})/(10**(-9))
SINR_k10_SN{} = (order_k10_SN{}_positive**2*p_k10_SN{})/((np.linalg.norm(w_k10_SN{}.conj().T.dot(h_k10_SN{})))**2*p_k10_SN{}+10**(-9))
SINR_k10_SN{} = (order_k10_SN{}_positive**2*p_k10_SN{})/((np.linalg.norm(w_k10_SN{}.conj().T.dot(h_k10_SN{})))**2*p_k10_SN{}+(np.linalg.norm(w_k10_SN{}.conj().T.dot(h_k10_SN{})))**2*p_k10_SN{}+10**(-9))'''.format(order_k10_index[2]+10,order_k10_index[2]+10,order_k10_index[2]+10,order_k10_index[1]+10,order_k10_index[1]+10,order_k10_index[1]+10,order_k10_index[1]+10,order_k10_index[2]+10,order_k10_index[2]+10,order_k10_index[0]+10,order_k10_index[0]+10,order_k10_index[0]+10,order_k10_index[0]+10,order_k10_index[2]+10,order_k10_index[2]+10,order_k10_index[0]+10,order_k10_index[1]+10,order_k10_index[1]+10))
                       
                        R_k10_SN10 = math.log((1+SINR_k10_SN10),2)
                        R_k10_SN11 = math.log((1+SINR_k10_SN11),2)
                        R_k10_SN12 = math.log((1+SINR_k10_SN12),2)
                        x_index=int(x_index*6+0.5)
                        y_index=int(y_index*6+0.5)+120
                        SINR_matrix_k7[y_index, x_index]=(h_k10_SN10[0]+h_k10_SN11[0]+h_k10_SN12[0]).real
                        # SINR_matrix_k7=SINR_matrix_k7.real
                ######################################
                # print('finish')
                plt.xlim(-10, 240)
                plt.ylim(-10, 240)
                if done7 or k7_done_label==1:
                    reward_k7 = 1
                # sns.heatmap(SINR_matrix_k7)
                # plt.show()
                
                ax = fig.add_subplot(111)
                im = ax.imshow(SINR_matrix_k7, cmap=plt.cm.viridis)
                # plt.show()
                # time.sleep(20) 
                xing_x = x_SN1*6
                xing_y = y_SN1*6
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                xing_x = x_SN2*6
                xing_y = y_SN2*6
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                xing_x = x_SN3*6
                xing_y = y_SN3*6
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                #######################################################################################################################
                xing_x = x_SN4*6+120
                xing_y = y_SN4*6
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                xing_x = x_SN5*6+120
                xing_y = y_SN5*6
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                xing_x = x_SN6*6+120
                xing_y = y_SN6*6
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                #############################################################################################################################
                xing_x = x_SN7*6+120
                xing_y = y_SN7*6+120
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                xing_x = x_SN8*6+120
                xing_y = y_SN8*6+120
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                xing_x = x_SN9*6+120
                xing_y = y_SN9*6+120
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                 #############################################################################################################################
                xing_x = x_SN10*6
                xing_y = y_SN10*6+120
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                xing_x = x_SN11*6
                xing_y = y_SN11*6+120
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                xing_x = x_SN12*6
                xing_y = y_SN12*6+120
                triangley = [xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y+(2.7)/2*math.sqrt(3)-1.6, xing_y-(2.7)/2*math.sqrt(3)-1.6]  # x的坐标
                trianglex = [xing_x-2.7, xing_x+2.7, xing_x]  # y的坐标
                plt.fill(trianglex, triangley, 'orange')
                
                triangleX = [xing_x+2.7, xing_x-2.7, xing_x]  # x的坐标
                triangleY = [xing_y-(2.7)/2*math.sqrt(3), xing_y-(2.7)/2*math.sqrt(3), xing_y+(2.7)/2*math.sqrt(3)]  # y的坐标
                plt.fill(triangleX, triangleY, 'orange')
                
                
                ax.add_patch(
                        pc.Rectangle(  # 长方形
                            (8*6, 8*6),  # （x,y）
                            3.5*6,  # 长
                            3.5*6,  # 宽
                            color='chocolate'  # 灰色
                        )
                    )
                
                ax.add_patch(
                        pc.Rectangle(  # 长方形
                            (28*6, 8*6),  # （x,y）
                            3.5*6,  # 长
                            3.5*6,  # 宽
                            color='chocolate'  # 灰色
                        )
                    )
                
                ax.add_patch(
                        pc.Rectangle(  # 长方形
                            (8*6, 28*6),  # （x,y）
                            3.5*6,  # 长
                            3.5*6,  # 宽
                            color='chocolate'  # 灰色
                        )
                    )
                
                ax.add_patch(
                        pc.Rectangle(  # 长方形
                            (28*6, 28*6),  # （x,y）
                            3.5*6,  # 长
                            3.5*6,  # 宽
                            color='chocolate'  # 灰色
                        )
                    )
                
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (0, 1*6),  # （x,y）
                            6,  # 长
                            6,  # 宽
                            color='w'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (18*6, 15*6),  # （x,y）
                            5,  # 长
                            5,  # 宽
                            color='violet'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (20*6, 1*6),  # （x,y）
                            6,  # 长
                            6,  # 宽
                            color='w'  # 浅紫色
                        )
                    )
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (38*6, 15*6),  # （x,y）
                            5,  # 长
                            5,  # 宽
                            color='violet'  # 浅紫色
                        )
                    )
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (20*6, 21*6),  # （x,y）
                            6,  # 长
                            6,  # 宽
                            color='w'  # 浅紫色
                        )
                    )
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (38*6, 35*6),  # （x,y）
                            5,  # 长
                            5,  # 宽
                            color='violet'  # 浅紫色
                        )
                    )
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (0*6, 21*6),  # （x,y）
                            6,  # 长
                            6,  # 宽
                            color='w'  # 浅紫色
                        )
                    )
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (18*6, 35*6),  # （x,y）
                            5,  # 长
                            5,  # 宽
                            color='violet'  # 浅紫色
                        )
                    )
                x_k1_array.append(x_1)
                y_k1_array.append(y_1)            
                for x_count in range(len(x_k1_array)):
        # for y_count in range(len(y_k1)):
                    ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_k1_array[x_count]*6, y_k1_array[x_count]*6),  # （x,y）
                            0.6*6,  # 长
                            0.6*6,  # 宽
                            color='red'  # 浅紫色
                        )
                    )
                x_k2_array.append(x_2)
                y_k2_array.append(y_2)            
                for x_count in range(len(x_k2_array)):
        # for y_count in range(len(y_k1)):
                    ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_k2_array[x_count]*6+120, y_k2_array[x_count]*6),  # （x,y）
                            0.6*6,  # 长
                            0.6*6,  # 宽
                            color='red'  # 浅紫色
                        )
                    )
                x_k7_array.append(x_7)
                y_k7_array.append(y_7)            
                for x_count in range(len(x_k7_array)):
        # for y_count in range(len(y_k1)):
                    ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_k7_array[x_count]*6+120, y_k7_array[x_count]*6+120),  # （x,y）
                            0.6*6,  # 长
                            0.6*6,  # 宽
                            color='red'  # 浅紫色
                        )
                    )
                x_k10_array.append(x_10)
                y_k10_array.append(y_10)            
                for x_count in range(len(x_k10_array)):
        # for y_count in range(len(y_k1)):
                    ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_k10_array[x_count]*6, y_k10_array[x_count]*6+120),  # （x,y）
                            0.6*6,  # 长
                            0.6*6,  # 宽
                            color='red'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_1*6, y_1*6),  # （x,y）
                            12,  # 长
                            9,  # 宽
                            color='lightgrey'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_1*6-2, y_1*6),  # （x,y）
                            1,  # 长
                            1,  # 宽
                            color='k'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_1*6+2, y_1*6),  # （x,y）
                            1,  # 长
                            1,  # 宽
                            color='k'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_2*6+120, y_2*6),  # （x,y）
                            12,  # 长
                            9,  # 宽
                            color='lightgrey'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_2*6-2+120, y_2*6),  # （x,y）
                            1,  # 长
                            1,  # 宽
                            color='k'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_2*6+2+120, y_2*6),  # （x,y）
                            1,  # 长
                            1,  # 宽
                            color='k'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_7*6+120, y_7*6+120),  # （x,y）
                            12,  # 长
                            9,  # 宽
                            color='lightgrey'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_7*6-2+120, y_7*6+120),  # （x,y）
                            1,  # 长
                            1,  # 宽
                            color='k'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_7*6+2+120, y_7*6+120),  # （x,y）
                            1,  # 长
                            1,  # 宽
                            color='k'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_10*6, y_10*6+120),  # （x,y）
                            12,  # 长
                            9,  # 宽
                            color='lightgrey'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_10*6-2, y_10*6+120),  # （x,y）
                            1,  # 长
                            1,  # 宽
                            color='k'  # 浅紫色
                        )
                    )
                
                ax.add_patch(
                        pc.Ellipse(  # 椭圆
                            (x_10*6+2, y_10*6+120),  # （x,y）
                            1,  # 长
                            1,  # 宽
                            color='k'  # 浅紫色
                        )
                    )
                
                plt.axis('off') 
                mkdir('{}'.format(episode))         
                plt.savefig('{}/picture_{}.png'.format(episode, h_count))
                plt.cla()
                # time.sleep(20) 
            # tu.setup(650,350,200,200)
            # tu.penup()
            # tu.fd(-250)
            # tu.pendown()
            # tu.pensize(25)
            # tu.seth(-40)
            # tu.bgpic('bg.png')
            # for i in range(4):
            #     tu.pencolor("yellow")
            #     tu.circle(40,80)
            #     tu.pencolor("gold")
            #     tu.circle(-40,80)
            # tu.circle(40,80/2)
            # tu.fd(40)
            # tu.circle(32,180)
            # tu.fd(40*2/3)
            # tu.done()
            
            
                # print('finish')
            # plt.cla()
            # RL take action and get next observation_k1_k1 and reward
            observation_k1_, reward_k1, done1 = env.step(action_k1)
            observation_k2_, reward_k2, done2 = env2.step(action_k2)
            #observation_k1_, reward, done = env.step(action)
            
            # reward_k1_reach = 0
            # if done:
            #     reward_k1_reach = 1
                
            # reward_k1 = ((0.0)*min((R_k1_SN1+R_k1_SN2+R_k1_SN3),100)*0.01+(1)*max(0, (d_k1_destination_max-d_k1_destination))/(d_k1_destination_max))*(0.2)+(1.5)*reward_k1_reach
            reward_k1 = (0.00005)*min((R_k1_SN1+R_k1_SN2+R_k1_SN3),50)
            reward_k2 = (0.00005)*min((R_k2_SN4+R_k2_SN5+R_k2_SN6),50)
            # reward_k1 = min(0.5, min(0, (d_k1_destination_max-d_k1_destination)/(d_k1_destination_max)*(0.5))+(0.00001)*min((R_k1_SN1+R_k1_SN2+R_k1_SN3),100))
            #if h_count<5000000:
            #print ('1',((- d_k1_destination)/(d_k1_destination_max))*(1), done)
            #print('2', (0.00005)*min((R_k1_SN1+R_k1_SN2+R_k1_SN3),100))
            
            if done1 or k1_done_label==1:
                reward_k1 = 1
                # k1_done_label = 1
            if done2 or k2_done_label==1:
                reward_k2 = 1
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
            
            
            
            
#             x_3 = observation_k3%xy_squre
#             y_3 = int(observation_k3/xy_squre)
#             # [x_SN1, y_SN1] = [0, 15]
#             # [x_SN2, y_SN2] = [5, 17]
#             # [x_SN3, y_SN3] = [13, 17]
#             # [x_SN4, y_SN4] = [3, 1]
#             # [x_SN5, y_SN5] = [11, 3]
#             # [x_SN6, y_SN6] = [15, 17]
            
#             d_k3_SN1 = (math.sqrt((x_3-x_SN1)*(x_3-x_SN1)+(y_3-y_SN1)*(y_3-y_SN1)))*Delta_xy+1
#             d_k3_SN2 = (math.sqrt((x_3-x_SN2)*(x_3-x_SN2)+(y_3-y_SN2)*(y_3-y_SN2)))*Delta_xy+1
#             d_k3_SN3 = (math.sqrt((x_3-x_SN3)*(x_3-x_SN3)+(y_3-y_SN3)*(y_3-y_SN3)))*Delta_xy+1
#             d_k3_SN4 = (math.sqrt((x_3-x_SN4-xy_squre)*(x_3-x_SN4-xy_squre)+(y_3-y_SN4)*(y_3-y_SN4)))*Delta_xy+1
#             d_k3_SN5 = (math.sqrt((x_3-x_SN5-xy_squre)*(x_3-x_SN5-xy_squre)+(y_3-y_SN5)*(y_3-y_SN5)))*Delta_xy+1
#             d_k3_SN6 = (math.sqrt((x_3-x_SN6-xy_squre)*(x_3-x_SN6-xy_squre)+(y_3-y_SN6)*(y_3-y_SN6)))*Delta_xy+1  
#             d_k3_destination = (math.sqrt((x_3-x_k3_destination)*(x_3-x_k3_destination)+(y_3-y_k3_destination)*(y_3-y_k3_destination)))*Delta_xy
#             beta_k3_SN1 = math.pow(10, -3)*math.pow(d_k3_SN1, 2.2)
#             beta_k3_SN2 = math.pow(10, -3)*math.pow(d_k3_SN2, 2.2)
#             beta_k3_SN3 = math.pow(10, -3)*math.pow(d_k3_SN3, 2.2) 
#             beta_k3_SN4 = math.pow(10, -3)*math.pow(d_k3_SN4, 2.2)
#             beta_k3_SN5 = math.pow(10, -3)*math.pow(d_k3_SN5, 2.2)
#             beta_k3_SN6 = math.pow(10, -3)*math.pow(d_k3_SN6, 2.2) 
#             theta_k3_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_3)/d_k3_SN1))
#             theta_k3_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_3)/d_k3_SN2))
#             theta_k3_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_3)/d_k3_SN3))
#             theta_k3_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4+xy_squre-x_3)/d_k3_SN4))
#             theta_k3_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5+xy_squre-x_3)/d_k3_SN5))
#             theta_k3_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6+xy_squre-x_3)/d_k3_SN6))
#             g_k3_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN1), math.exp(1)**(-1*1j*theta_k3_SN1*2)])
#             g_k3_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN2), math.exp(1)**(-1*1j*theta_k3_SN2*2)])
#             g_k3_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN3), math.exp(1)**(-1*1j*theta_k3_SN3*2)])
#             g_k3_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN4), math.exp(1)**(-1*1j*theta_k3_SN4*2)])
#             g_k3_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN5), math.exp(1)**(-1*1j*theta_k3_SN5*2)])
#             g_k3_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN6), math.exp(1)**(-1*1j*theta_k3_SN6*2)])
#             g_k3_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN1 = g_k3_SN1_LOS + g_k3_SN1_NLOS
#             g_k3_SN2 = g_k3_SN2_LOS + g_k3_SN2_NLOS
#             g_k3_SN3 = g_k3_SN3_LOS + g_k3_SN3_NLOS
#             g_k3_SN4 = g_k3_SN4_LOS + g_k3_SN4_NLOS
#             g_k3_SN5 = g_k3_SN5_LOS + g_k3_SN5_NLOS
#             g_k3_SN6 = g_k3_SN6_LOS + g_k3_SN6_NLOS
#             h_k3_SN1 = math.sqrt(beta_k3_SN1)*g_k3_SN1
#             h_k3_SN2 = math.sqrt(beta_k3_SN2)*g_k3_SN2
#             h_k3_SN3 = math.sqrt(beta_k3_SN3)*g_k3_SN3
#             h_k3_SN4 = math.sqrt(beta_k3_SN4)*g_k3_SN4            
#             h_k3_SN5 = math.sqrt(beta_k3_SN5)*g_k3_SN5
#             h_k3_SN6 = math.sqrt(beta_k3_SN6)*g_k3_SN6
#             #communication model
#             d_k3_SN7 = (math.sqrt((x_3-x_SN7-xy_squre)*(x_3-x_SN7-xy_squre)+(y_3-y_SN7-xy_squre)*(y_3-y_SN7-xy_squre)))*Delta_xy+1
#             d_k3_SN8 = (math.sqrt((x_3-x_SN8-xy_squre)*(x_3-x_SN8-xy_squre)+(y_3-y_SN8-xy_squre)*(y_3-y_SN8-xy_squre)))*Delta_xy+1
#             d_k3_SN9 = (math.sqrt((x_3-x_SN9-xy_squre)*(x_3-x_SN9-xy_squre)+(y_3-y_SN9-xy_squre)*(y_3-y_SN9-xy_squre)))*Delta_xy+1
#             beta_k3_SN7 = math.pow(10, -3)*math.pow(d_k3_SN7, 2.2)
#             beta_k3_SN8 = math.pow(10, -3)*math.pow(d_k3_SN8, 2.2)
#             beta_k3_SN9 = math.pow(10, -3)*math.pow(d_k3_SN9, 2.2) 
#             theta_k3_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_3)/d_k3_SN7))
#             theta_k3_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_3)/d_k3_SN8))
#             theta_k3_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_3)/d_k3_SN9))
#             g_k3_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN7), math.exp(1)**(-1*1j*theta_k3_SN7*2)])
#             g_k3_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN8), math.exp(1)**(-1*1j*theta_k3_SN8*2)])
#             g_k3_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k3_SN9), math.exp(1)**(-1*1j*theta_k3_SN9*2)])
#             g_k3_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k3_SN7 = g_k3_SN7_LOS + g_k3_SN7_NLOS
#             g_k3_SN8 = g_k3_SN8_LOS + g_k3_SN8_NLOS
#             g_k3_SN9 = g_k3_SN9_LOS + g_k3_SN9_NLOS
#             h_k3_SN7 = math.sqrt(beta_k3_SN7)*g_k3_SN7            
#             h_k3_SN8 = math.sqrt(beta_k3_SN8)*g_k3_SN8
#             h_k3_SN9 = math.sqrt(beta_k3_SN9)*g_k3_SN9
#             H_k3_PART3 = np.array([h_k3_SN7[0],h_k3_SN7[1],h_k3_SN7[2],h_k3_SN8[0],h_k3_SN8[1],h_k3_SN8[2],h_k3_SN9[0],h_k3_SN9[1],h_k3_SN9[2]]).T
#             H_k3_PART1 = np.array([h_k3_SN1[0],h_k3_SN1[1],h_k3_SN1[2],h_k3_SN2[0],h_k3_SN2[1],h_k3_SN2[2],h_k3_SN3[0],h_k3_SN3[1],h_k3_SN3[2]]).T
#             H_k3_PART2 = np.array([h_k3_SN4[0],h_k3_SN4[1],h_k3_SN4[2],h_k3_SN5[0],h_k3_SN5[1],h_k3_SN5[2],h_k3_SN6[0],h_k3_SN6[1],h_k3_SN6[2]]).T
    
    
#             #communication model
#             H_k3 = np.array([H_k3_PART1, H_k3_PART2,  H_k3_PART3]).T
#             # try:
#             #     W_bar_k3 = H_k3.dot(lg.inv(H_k3.conj().T.dot(H_k3)))
#             # except:
#             #     W_bar_k3 = W_bar_k3
#             # else:
#             #     W_bar_k3 = H_k3.dot(lg.inv(H_k3.conj().T.dot(H_k3)))
#             W_bar_k3 = H_k3.dot(lg.inv(H_k3.conj().T.dot(H_k3)))
#             w_k3_k3 = W_bar_k3[:, 0]/(np.linalg.norm(W_bar_k3[:, 0]))
#             # w_k3_k4 = W_bar_k3[:, 1]/(np.linalg.norm(W_bar_k3[:, 1]))
#             # w_k3_SN3 = W_bar_k3[:, 2]/(np.linalg.norm(W_bar_k3[:, 2]))
#             # w_k3_SN4 = W_bar_k3[:, 3]/(np.linalg.norm(W_bar_k3[:, 3]))
#             # w_k3_SN5 = W_bar_k3[:, 4]/(np.linalg.norm(W_bar_k3[:, 4]))
#             # w_k3_SN6 = W_bar_k3[:, 5]/(np.linalg.norm(W_bar_k3[:, 5]))
#             # w_check3 = np.linalg.norm(w_k3_SN1)
#             # w_check4 = np.linalg.norm(w_k3_SN2)
#             # w_check3 = np.linalg.norm(w_k3_SN3)
#             # w_k3_SN6=w_k3_k4[4:6]
#             # w_k3_SN5=w_k3_k4[2:4]
#             # w_k3_SN4=w_k3_k4[0:2]
#             w_k3_SN3=w_k3_k3[6:9]
#             w_k3_SN2=w_k3_k3[3:6]
#             w_k3_SN1=w_k3_k3[0:3]
#             # w_check3 = np.linalg.norm(w_k3_SN1)
#             # w_check4 = np.linalg.norm(w_k3_SN2)
#             # w_check3 = np.linalg.norm(w_k3_SN3)
#             order_k3_SN1 = w_k3_SN1.conj().T.dot(h_k3_SN1)
#             order_k3_SN2 = w_k3_SN2.conj().T.dot(h_k3_SN2)
#             order_k3_SN3 = w_k3_SN3.conj().T.dot(h_k3_SN3)
#             order_k3_SN1_positive = np.linalg.norm(order_k3_SN1)
#             order_k3_SN2_positive = np.linalg.norm(order_k3_SN2)
#             order_k3_SN3_positive = np.linalg.norm(order_k3_SN3)
#             # print(order_k3_SN1_positive, order_k3_SN2_positive, order_k3_SN3_positive)
    
#             x_4 = observation_k4%xy_squre
#             y_4 = int(observation_k4/xy_squre)
#             d_k4_SN4 = (math.sqrt((x_4-x_SN4)*(x_4-x_SN4)+(y_4-y_SN4)*(y_4-y_SN4)))*Delta_xy+1
#             d_k4_SN5 = (math.sqrt((x_4-x_SN5)*(x_4-x_SN5)+(y_4-y_SN5)*(y_4-y_SN5)))*Delta_xy+1
#             d_k4_SN6 = (math.sqrt((x_4-x_SN6)*(x_4-x_SN6)+(y_4-y_SN6)*(y_4-y_SN6)))*Delta_xy+1
#             d_k4_SN1 = (math.sqrt((x_4+xy_squre-x_SN1)*(x_4+xy_squre-x_SN1)+(y_4-y_SN1)*(y_4-y_SN1)))*Delta_xy+1
#             d_k4_SN2 = (math.sqrt((x_4+xy_squre-x_SN2)*(x_4+xy_squre-x_SN2)+(y_4-y_SN2)*(y_4-y_SN2)))*Delta_xy+1
#             d_k4_SN3 = (math.sqrt((x_4+xy_squre-x_SN3)*(x_4+xy_squre-x_SN3)+(y_4-y_SN3)*(y_4-y_SN3)))*Delta_xy+1
#             d_k4_destination = (math.sqrt((x_4-x_k4_destination)*(x_4-x_k4_destination)+(y_4-y_k4_destination)*(y_4-y_k4_destination)))*Delta_xy
#             beta_k4_SN4 = math.pow(10, -3)*math.pow(d_k4_SN4, 2.2)
#             beta_k4_SN5 = math.pow(10, -3)*math.pow(d_k4_SN5, 2.2)
#             beta_k4_SN6 = math.pow(10, -3)*math.pow(d_k4_SN6, 2.2)
#             beta_k4_SN1 = math.pow(10, -3)*math.pow(d_k4_SN1, 2.2)
#             beta_k4_SN2 = math.pow(10, -3)*math.pow(d_k4_SN2, 2.2)
#             beta_k4_SN3 = math.pow(10, -3)*math.pow(d_k4_SN3, 2.2)
#             theta_k4_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_4)/d_k4_SN4))
#             theta_k4_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_4)/d_k4_SN5))
#             theta_k4_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_4)/d_k4_SN6))
#             theta_k4_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_4-xy_squre)/d_k4_SN1))
#             theta_k4_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_4-xy_squre)/d_k4_SN2))
#             theta_k4_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_4-xy_squre)/d_k4_SN3))
#             g_k4_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN1), math.exp(1)**(-1*1j*theta_k4_SN1*2)])
#             g_k4_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN2), math.exp(1)**(-1*1j*theta_k4_SN2*2)])
#             g_k4_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN3), math.exp(1)**(-1*1j*theta_k4_SN3*2)])
#             g_k4_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN4), math.exp(1)**(-1*1j*theta_k4_SN4*2)])
#             g_k4_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN5), math.exp(1)**(-1*1j*theta_k4_SN5*2)])
#             g_k4_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN6), math.exp(1)**(-1*1j*theta_k4_SN6*2)])
#             g_k4_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN4 = g_k4_SN4_LOS + g_k4_SN4_NLOS
#             g_k4_SN5 = g_k4_SN5_LOS + g_k4_SN5_NLOS
#             g_k4_SN6 = g_k4_SN6_LOS + g_k4_SN6_NLOS
#             g_k4_SN1 = g_k4_SN1_LOS + g_k4_SN1_NLOS
#             g_k4_SN2 = g_k4_SN2_LOS + g_k4_SN2_NLOS
#             g_k4_SN3 = g_k4_SN3_LOS + g_k4_SN3_NLOS
#             h_k4_SN4 = math.sqrt(beta_k4_SN4)*g_k4_SN4            
#             h_k4_SN5 = math.sqrt(beta_k4_SN5)*g_k4_SN5
#             h_k4_SN6 = math.sqrt(beta_k4_SN6)*g_k4_SN6
#             h_k4_SN1 = math.sqrt(beta_k4_SN1)*g_k4_SN1            
#             h_k4_SN2 = math.sqrt(beta_k4_SN2)*g_k4_SN2
#             h_k4_SN3 = math.sqrt(beta_k4_SN3)*g_k4_SN3
#             #communication model
#             d_k4_SN7 = (math.sqrt((x_4-x_SN7)*(x_4-x_SN7)+(y_4-y_SN7-xy_squre)*(y_4-y_SN7-xy_squre)))*Delta_xy+1
#             d_k4_SN8 = (math.sqrt((x_4-x_SN8)*(x_4-x_SN8)+(y_4-y_SN8-xy_squre)*(y_4-y_SN8-xy_squre)))*Delta_xy+1
#             d_k4_SN9 = (math.sqrt((x_4-x_SN9)*(x_4-x_SN9)+(y_4-y_SN9-xy_squre)*(y_4-y_SN9-xy_squre)))*Delta_xy+1
#             beta_k4_SN7 = math.pow(10, -3)*math.pow(d_k4_SN7, 2.2)
#             beta_k4_SN8 = math.pow(10, -3)*math.pow(d_k4_SN8, 2.2)
#             beta_k4_SN9 = math.pow(10, -3)*math.pow(d_k4_SN9, 2.2) 
#             theta_k4_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_4)/d_k4_SN7))
#             theta_k4_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_4)/d_k4_SN8))
#             theta_k4_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_4)/d_k4_SN9))
#             g_k4_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN7), math.exp(1)**(-1*1j*theta_k4_SN7*2)])
#             g_k4_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN8), math.exp(1)**(-1*1j*theta_k4_SN8*2)])
#             g_k4_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k4_SN9), math.exp(1)**(-1*1j*theta_k4_SN9*2)])
#             g_k4_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k4_SN7 = g_k4_SN7_LOS + g_k4_SN7_NLOS
#             g_k4_SN8 = g_k4_SN8_LOS + g_k4_SN8_NLOS
#             g_k4_SN9 = g_k4_SN9_LOS + g_k4_SN9_NLOS
#             h_k4_SN7 = math.sqrt(beta_k4_SN7)*g_k4_SN7            
#             h_k4_SN8 = math.sqrt(beta_k4_SN8)*g_k4_SN8
#             h_k4_SN9 = math.sqrt(beta_k4_SN9)*g_k4_SN9
#             H_k4_PART3 = np.array([h_k4_SN7[0],h_k4_SN7[1],h_k4_SN7[2],h_k4_SN8[0],h_k4_SN8[1],h_k4_SN8[2],h_k4_SN9[0],h_k4_SN9[1],h_k4_SN9[2]]).T
#             H_k4_PART1 = np.array([h_k4_SN1[0],h_k4_SN1[1],h_k4_SN1[2],h_k4_SN2[0],h_k4_SN2[1],h_k4_SN2[2],h_k4_SN3[0],h_k4_SN3[1],h_k4_SN3[2]]).T
#             H_k4_PART2 = np.array([h_k4_SN4[0],h_k4_SN4[1],h_k4_SN4[2],h_k4_SN5[0],h_k4_SN5[1],h_k4_SN5[2],h_k4_SN6[0],h_k4_SN6[1],h_k4_SN6[2]]).T

            
#             #communication model
#             H_k4 = np.array([H_k4_PART1, H_k4_PART2, H_k4_PART3]).T
#             # try:
#             #     W_bar_k4 = H_k4.dot(lg.inv(H_k4.conj().T.dot(H_k4)))
#             # except:
#             #     W_bar_k4 = W_bar_k4
#             # else:
#             #     W_bar_k4 = H_k4.dot(lg.inv(H_k4.conj().T.dot(H_k4)))
#             W_bar_k4 = H_k4.dot(lg.inv(H_k4.conj().T.dot(H_k4)))
#             # w_k4_k3 = W_bar_k4[:, 0]/(np.linalg.norm(W_bar_k4[:, 0]))
#             w_k4_k4 = W_bar_k4[:, 1]/(np.linalg.norm(W_bar_k4[:, 1]))
#             # w_k4_SN6 = W_bar_k4[:, 2]/(np.linalg.norm(W_bar_k4[:, 2]))
#             # w_k4_SN1 = W_bar_k4[:, 3]/(np.linalg.norm(W_bar_k4[:, 3]))
#             # w_k4_SN2 = W_bar_k4[:, 4]/(np.linalg.norm(W_bar_k4[:, 4]))
#             # w_k4_SN3 = W_bar_k4[:, 5]/(np.linalg.norm(W_bar_k4[:, 5]))
#             # w_check3 = np.linalg.norm(w_k3_SN4)
#             # w_check4 = np.linalg.norm(w_k3_SN5)
#             # w_check3 = np.linalg.norm(w_k3_SN6)
#             w_k4_SN6=w_k4_k4[6:9]
#             w_k4_SN5=w_k4_k4[3:6]
#             w_k4_SN4=w_k4_k4[0:3]
#             # w_k4_SN3=w_k4_k3[4:6]
#             # w_k4_SN2=w_k4_k3[2:4]
#             # w_k4_SN1=w_k4_k3[0:2]
#             # w_check3 = np.linalg.norm(w_k3_SN4)
#             # w_check4 = np.linalg.norm(w_k3_SN5)
#             # w_check3 = np.linalg.norm(w_k3_SN6)
#             order_k4_SN4 = w_k4_SN4.conj().T.dot(h_k4_SN4)
#             order_k4_SN5 = w_k4_SN5.conj().T.dot(h_k4_SN5)
#             order_k4_SN6 = w_k4_SN6.conj().T.dot(h_k4_SN6)
#             order_k4_SN4_positive = np.linalg.norm(order_k4_SN4)
#             order_k4_SN5_positive = np.linalg.norm(order_k4_SN5)
#             order_k4_SN6_positive = np.linalg.norm(order_k4_SN6)
#             # print(order_k3_SN1_positive, order_k3_SN5_positive, order_k3_SN3_positive)
    
#             x_7 = observation_k8%xy_squre
#             y_7 = int(observation_k8/xy_squre)
#             d_k8_SN4 = (math.sqrt((x_7-x_SN4)*(x_7-x_SN4)+(y_7+xy_squre-y_SN4)*(y_7+xy_squre-y_SN4)))*Delta_xy+1
#             d_k8_SN5 = (math.sqrt((x_7-x_SN5)*(x_7-x_SN5)+(y_7+xy_squre-y_SN5)*(y_7+xy_squre-y_SN5)))*Delta_xy+1
#             d_k8_SN6 = (math.sqrt((x_7-x_SN6)*(x_7-x_SN6)+(y_7+xy_squre-y_SN6)*(y_7+xy_squre-y_SN6)))*Delta_xy+1
#             d_k8_SN1 = (math.sqrt((x_7+xy_squre-x_SN1)*(x_7+xy_squre-x_SN1)+(y_7+xy_squre-y_SN1)*(y_7+xy_squre-y_SN1)))*Delta_xy+1
#             d_k8_SN2 = (math.sqrt((x_7+xy_squre-x_SN2)*(x_7+xy_squre-x_SN2)+(y_7+xy_squre-y_SN2)*(y_7+xy_squre-y_SN2)))*Delta_xy+1
#             d_k8_SN3 = (math.sqrt((x_7+xy_squre-x_SN3)*(x_7+xy_squre-x_SN3)+(y_7+xy_squre-y_SN3)*(y_7+xy_squre-y_SN3)))*Delta_xy+1
#             d_k8_destination = (math.sqrt((x_7-x_k8_destination)*(x_7-x_k8_destination)+(y_7-y_k8_destination)*(y_7-y_k8_destination)))*Delta_xy
#             beta_k8_SN4 = math.pow(10, -3)*math.pow(d_k8_SN4, 2.2)
#             beta_k8_SN5 = math.pow(10, -3)*math.pow(d_k8_SN5, 2.2)
#             beta_k8_SN6 = math.pow(10, -3)*math.pow(d_k8_SN6, 2.2)
#             beta_k8_SN1 = math.pow(10, -3)*math.pow(d_k8_SN1, 2.2)
#             beta_k8_SN2 = math.pow(10, -3)*math.pow(d_k8_SN2, 2.2)
#             beta_k8_SN3 = math.pow(10, -3)*math.pow(d_k8_SN3, 2.2)
#             theta_k8_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_7)/d_k8_SN4))
#             theta_k8_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_7)/d_k8_SN5))
#             theta_k8_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_7)/d_k8_SN6))
#             theta_k8_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_7-xy_squre)/d_k8_SN1))
#             theta_k8_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_7-xy_squre)/d_k8_SN2))
#             theta_k8_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_7-xy_squre)/d_k8_SN3))
#             g_k8_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN4), math.exp(1)**(-1*1j*theta_k8_SN4*2)])
#             g_k8_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN5), math.exp(1)**(-1*1j*theta_k8_SN5*2)])
#             g_k8_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN6), math.exp(1)**(-1*1j*theta_k8_SN6*2)])
#             g_k8_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN1), math.exp(1)**(-1*1j*theta_k8_SN1*2)])
#             g_k8_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN2), math.exp(1)**(-1*1j*theta_k8_SN2*2)])
#             g_k8_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN3), math.exp(1)**(-1*1j*theta_k8_SN3*2)])
#             g_k8_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN4 = g_k8_SN4_LOS + g_k8_SN4_NLOS
#             g_k8_SN5 = g_k8_SN5_LOS + g_k8_SN5_NLOS
#             g_k8_SN6 = g_k8_SN6_LOS + g_k8_SN6_NLOS
#             g_k8_SN1 = g_k8_SN1_LOS + g_k8_SN1_NLOS
#             g_k8_SN2 = g_k8_SN2_LOS + g_k8_SN2_NLOS
#             g_k8_SN3 = g_k8_SN3_LOS + g_k8_SN3_NLOS
#             h_k8_SN4 = math.sqrt(beta_k8_SN4)*g_k8_SN4            
#             h_k8_SN5 = math.sqrt(beta_k8_SN5)*g_k8_SN5
#             h_k8_SN6 = math.sqrt(beta_k8_SN6)*g_k8_SN6
#             h_k8_SN1 = math.sqrt(beta_k8_SN1)*g_k8_SN1            
#             h_k8_SN2 = math.sqrt(beta_k8_SN2)*g_k8_SN2
#             h_k8_SN3 = math.sqrt(beta_k8_SN3)*g_k8_SN3
            
#             d_k8_SN7 = (math.sqrt((x_8-x_SN7)*(x_8-x_SN7)+(y_8-y_SN7)*(y_8-y_SN7)))*Delta_xy+1
#             d_k8_SN8 = (math.sqrt((x_8-x_SN8)*(x_8-x_SN8)+(y_8-y_SN8)*(y_8-y_SN8)))*Delta_xy+1
#             d_k8_SN9 = (math.sqrt((x_8-x_SN9)*(x_8-x_SN9)+(y_8-y_SN9)*(y_8-y_SN9)))*Delta_xy+1
#             beta_k8_SN7 = math.pow(10, -3)*math.pow(d_k8_SN7, 2.2)
#             beta_k8_SN8 = math.pow(10, -3)*math.pow(d_k8_SN8, 2.2)
#             beta_k8_SN9 = math.pow(10, -3)*math.pow(d_k8_SN9, 2.2) 
#             theta_k8_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_8)/d_k8_SN7))
#             theta_k8_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_8)/d_k8_SN8))
#             theta_k8_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_8)/d_k8_SN9))
#             g_k8_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN7), math.exp(1)**(-1*1j*theta_k8_SN7*2)])
#             g_k8_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN8), math.exp(1)**(-1*1j*theta_k8_SN8*2)])
#             g_k8_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k8_SN9), math.exp(1)**(-1*1j*theta_k8_SN9*2)])
#             g_k8_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k8_SN7 = g_k8_SN7_LOS + g_k8_SN7_NLOS
#             g_k8_SN8 = g_k8_SN8_LOS + g_k8_SN8_NLOS
#             g_k8_SN9 = g_k8_SN9_LOS + g_k8_SN9_NLOS
#             h_k8_SN7 = math.sqrt(beta_k8_SN7)*g_k8_SN7            
#             h_k8_SN8 = math.sqrt(beta_k8_SN8)*g_k8_SN8
#             h_k8_SN9 = math.sqrt(beta_k8_SN9)*g_k8_SN9
#             H_k8_PART3 = np.array([h_k8_SN7[0],h_k8_SN7[1],h_k8_SN7[2],h_k8_SN8[0],h_k8_SN8[1],h_k8_SN8[2],h_k8_SN9[0],h_k8_SN9[1],h_k8_SN9[2]]).T
#             H_k8_PART1 = np.array([h_k8_SN1[0],h_k8_SN1[1],h_k8_SN1[2],h_k8_SN2[0],h_k8_SN2[1],h_k8_SN2[2],h_k8_SN3[0],h_k8_SN3[1],h_k8_SN3[2]]).T
#             H_k8_PART2 = np.array([h_k8_SN4[0],h_k8_SN4[1],h_k8_SN4[2],h_k8_SN5[0],h_k8_SN5[1],h_k8_SN5[2],h_k8_SN6[0],h_k8_SN6[1],h_k8_SN6[2]]).T

            
#             #communication model
#             H_k8 = np.array([H_k8_PART1, H_k8_PART2, H_k8_PART3]).T
#             # try:
#             #     W_bar_k8 = H_k8.dot(lg.inv(H_k8.conj().T.dot(H_k8)))
#             # except:
#             #     W_bar_k7 = W_bar_k7
#             # else:
#             #     W_bar_k7 = H_k7.dot(lg.inv(H_k7.conj().T.dot(H_k7)))
#             W_bar_k8 = H_k8.dot(lg.inv(H_k8.conj().T.dot(H_k8)))
#             # w_k8_k1 = W_bar_k8[:, 0]/(np.linalg.norm(W_bar_k8[:, 0]))
#             w_k8_k8 = W_bar_k8[:, 2]/(np.linalg.norm(W_bar_k8[:, 2]))
#             # w_k7_SN6 = W_bar_k7[:, 2]/(np.linalg.norm(W_bar_k7[:, 2]))
#             # w_k7_SN1 = W_bar_k7[:, 3]/(np.linalg.norm(W_bar_k7[:, 3]))
#             # w_k7_SN2 = W_bar_k7[:, 4]/(np.linalg.norm(W_bar_k7[:, 4]))
#             # w_k7_SN3 = W_bar_k7[:, 5]/(np.linalg.norm(W_bar_k7[:, 5]))
#             # w_check1 = np.linalg.norm(w_k1_SN4)
#             # w_check7 = np.linalg.norm(w_k1_SN5)
#             # w_check3 = np.linalg.norm(w_k1_SN6)
#             w_k8_SN9=w_k8_k8[6:9]
#             w_k8_SN8=w_k8_k8[3:6]
#             w_k8_SN7=w_k8_k8[0:3]
#             # w_k7_SN3=w_k7_k1[4:6]
#             # w_k7_SN2=w_k7_k1[2:4]
#             # w_k7_SN1=w_k7_k1[0:2]
#             # w_check1 = np.linalg.norm(w_k1_SN4)
#             # w_check7 = np.linalg.norm(w_k1_SN5)
#             # w_check3 = np.linalg.norm(w_k1_SN6)
#             order_k8_SN7 = w_k8_SN7.conj().T.dot(h_k8_SN7)
#             order_k8_SN8 = w_k8_SN8.conj().T.dot(h_k8_SN8)
#             order_k8_SN9 = w_k8_SN9.conj().T.dot(h_k8_SN9)
#             order_k8_SN7_positive = np.linalg.norm(order_k8_SN7)
#             order_k8_SN8_positive = np.linalg.norm(order_k8_SN8)
#             order_k8_SN9_positive = np.linalg.norm(order_k8_SN9)
    
    
#             # observation_k32 = observation_k3 + observation_k4*xy_squre*xy_squre
#             # RL choose action based on observation_k3
#             if k3_done_label == 0:
#                 action_k3 = RL3.choose_action(observation_k3)
#             if k4_done_label == 0:
#                 action_k4 = RL4.choose_action(observation_k4)
#             # action_k3 = action_k32%ACTION_SHAPE_single
#             # action_k4 = int(action_k32/ACTION_SHAPE_single)
#             order_k3 = [order_k3_SN1_positive, order_k3_SN2_positive, order_k3_SN3_positive]
#             order_k3_index = [order_k3[0] for order_k3 in sorted(enumerate(order_k3),key=lambda i:i[1], reverse=True)]
#             p_k3_SN_index = M_k
#             for i in order_k3_index:
#                 exec('''p_k3_SN{}_int = p_c63[int(action_k3/5), p_k3_SN_index-1]
# p_k3_SN{} = p_k3_SN{}_int*P_SN_mav/(6+1)
# p_k3_SN_index = p_k3_SN_index-1'''.format(i+1, i+1, i+1))
            
#             order_k3_index = [order_k3[0] for order_k3 in sorted(enumerate(order_k3),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k3_SN{} = (order_k3_SN{}_positive**2*p_k3_SN{})/(10**(-9))
# SINR_k3_SN{} = (order_k3_SN{}_positive**2*p_k3_SN{})/((np.linalg.norm(w_k3_SN{}.conj().T.dot(h_k3_SN{})))**2*p_k3_SN{}+10**(-9))
# SINR_k3_SN{} = (order_k3_SN{}_positive**2*p_k3_SN{})/((np.linalg.norm(w_k3_SN{}.conj().T.dot(h_k3_SN{})))**2*p_k3_SN{}+(np.linalg.norm(w_k3_SN{}.conj().T.dot(h_k3_SN{})))**2*p_k3_SN{}+10**(-9))'''.format(order_k3_index[2]+1,order_k3_index[2]+1,order_k3_index[2]+1,order_k3_index[1]+1,order_k3_index[1]+1,order_k3_index[1]+1,order_k3_index[1]+1,order_k3_index[2]+1,order_k3_index[2]+1,order_k3_index[0]+1,order_k3_index[0]+1,order_k3_index[0]+1,order_k3_index[0]+1,order_k3_index[2]+1,order_k3_index[2]+1,order_k3_index[0]+1,order_k3_index[1]+1,order_k3_index[1]+1))
    
#             order_k4 = [order_k4_SN4_positive, order_k4_SN5_positive, order_k4_SN6_positive]
#             order_k4_index = [order_k4[0] for order_k4 in sorted(enumerate(order_k4),key=lambda i:i[1], reverse=True)]
#             p_k4_SN_index = M_k
#             for i in order_k4_index:
#                 exec('''p_k4_SN{}_int = p_c63[int(action_k4/5), p_k4_SN_index-1]
# p_k4_SN{} = p_k4_SN{}_int*P_SN_mav/(6+1)
# p_k4_SN_index = p_k4_SN_index-1'''.format(i+4, i+4, i+4))
            
#             order_k4_index = [order_k4[0] for order_k4 in sorted(enumerate(order_k4),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k4_SN{} = (order_k4_SN{}_positive**2*p_k4_SN{})/(10**(-9))
# SINR_k4_SN{} = (order_k4_SN{}_positive**2*p_k4_SN{})/((np.linalg.norm(w_k4_SN{}.conj().T.dot(h_k4_SN{})))**2*p_k4_SN{}+10**(-9))
# SINR_k4_SN{} = (order_k4_SN{}_positive**2*p_k4_SN{})/((np.linalg.norm(w_k4_SN{}.conj().T.dot(h_k4_SN{})))**2*p_k4_SN{}+(np.linalg.norm(w_k4_SN{}.conj().T.dot(h_k4_SN{})))**2*p_k4_SN{}+10**(-9))'''.format(order_k4_index[2]+4,order_k4_index[2]+4,order_k4_index[2]+4,order_k4_index[1]+4,order_k4_index[1]+4,order_k4_index[1]+4,order_k4_index[1]+4,order_k4_index[2]+4,order_k4_index[2]+4,order_k4_index[0]+4,order_k4_index[0]+4,order_k4_index[0]+4,order_k4_index[0]+4,order_k4_index[2]+4,order_k4_index[2]+4,order_k4_index[0]+4,order_k4_index[1]+4,order_k4_index[1]+4))
            
#             # print(np.linalg.norm(w_k4_SN4.conj().T.dot(h_k4_SN2)), np.linalg.norm(w_k4_SN5.conj().T.dot(h_k4_SN4)), np.linalg.norm(w_k4_SN4.conj().T.dot(h_k4_SN1)), np.linalg.norm(w_k4_SN6.conj().T.dot(h_k4_SN4)))
#             # print(np.linalg.norm(w_k3_SN1.conj().T.dot(h_k3_SN2)), np.linalg.norm(w_k3_SN1.conj().T.dot(h_k3_SN4)), np.linalg.norm(w_k3_SN2.conj().T.dot(h_k3_SN5)), np.linalg.norm(w_k3_SN2.conj().T.dot(h_k3_SN4)))
#             if k8_done_label == 0:
#                 action_k8 = RL8.choose_action(observation_k8)
#             # action_k1 = action_k12%ACTION_SHAPE_single
#             # action_k2 = int(action_k12/ACTION_SHAPE_single)
#             order_k8 = [order_k8_SN7_positive, order_k8_SN8_positive, order_k8_SN9_positive]
#             order_k8_index = [order_k8[0] for order_k8 in sorted(enumerate(order_k8),key=lambda i:i[1], reverse=True)]
#             p_k8_SN_index = M_k
#             for i in order_k8_index:
#                 exec('''p_k8_SN{}_int = p_c63[int(action_k8/5), p_k8_SN_index-1]
# p_k8_SN{} = p_k8_SN{}_int*P_SN_mav/(6+1)
# p_k8_SN_index = p_k8_SN_index-1'''.format(i+7, i+7, i+7))
            
#             order_k8_index = [order_k8[0] for order_k8 in sorted(enumerate(order_k8),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k8_SN{} = (order_k8_SN{}_positive**2*p_k8_SN{})/(10**(-9))
# SINR_k8_SN{} = (order_k8_SN{}_positive**2*p_k8_SN{})/((np.linalg.norm(w_k8_SN{}.conj().T.dot(h_k8_SN{})))**2*p_k8_SN{}+70**(-9))
# SINR_k8_SN{} = (order_k8_SN{}_positive**2*p_k8_SN{})/((np.linalg.norm(w_k8_SN{}.conj().T.dot(h_k8_SN{})))**2*p_k8_SN{}+(np.linalg.norm(w_k8_SN{}.conj().T.dot(h_k8_SN{})))**2*p_k8_SN{}+10**(-9))'''.format(order_k8_index[2]+7,order_k8_index[2]+7,order_k8_index[2]+7,order_k8_index[1]+7,order_k8_index[1]+7,order_k8_index[1]+7,order_k8_index[1]+7,order_k8_index[2]+7,order_k8_index[2]+7,order_k8_index[0]+7,order_k8_index[0]+7,order_k8_index[0]+7,order_k8_index[0]+7,order_k8_index[2]+7,order_k8_index[2]+7,order_k8_index[0]+7,order_k8_index[1]+7,order_k8_index[1]+7))
#             R_k8_SN7 = math.log((1+SINR_k8_SN7),2)
#             R_k8_SN8 = math.log((1+SINR_k8_SN8),2)
#             R_k8_SN9 = math.log((1+SINR_k8_SN9),2)
#             R_k3_SN1 = math.log((1+SINR_k3_SN1),2)
#             R_k3_SN2 = math.log((1+SINR_k3_SN2),2)
#             R_k3_SN3 = math.log((1+SINR_k3_SN3),2)
#             R_k4_SN4 = math.log((1+SINR_k4_SN4),2)
#             R_k4_SN5 = math.log((1+SINR_k4_SN5),2)
#             R_k4_SN6 = math.log((1+SINR_k4_SN6),2)
    
#             # RL take action and get next observation_k3_k3 and reward
#             observation_k3_, reward_k3, done3 = env3.step(action_k3)
#             observation_k4_, reward_k4, done4 = env4.step(action_k4)
#             #observation_k3_, reward, done = env.step(action)
#             observation_k8_, reward_k8, done8 = env8.step(action_k8)
#             reward_k8 = (0.00005)*min((R_k8_SN7+R_k8_SN8+R_k8_SN9),50)
#             if done8 or k8_done_label==1:
#                 reward_k8 = 1
#             # reward_k3_reach = 0
#             # if done:
#             #     reward_k3_reach = 1

#             # reward_k3 = ((0.0)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),100)*0.01+(1)*max(0, (d_k3_destination_max-d_k3_destination))/(d_k3_destination_max))*(0.2)+(1.5)*reward_k3_reach
#             reward_k3 = (0.00005)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),50)
#             reward_k4 = (0.00005)*min((R_k4_SN4+R_k4_SN5+R_k4_SN6),50)
#             # reward_k3 = min(0.5, min(0, (d_k3_destination_max-d_k3_destination)/(d_k3_destination_max)*(0.5))+(0.00001)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),100))
#             #if h_count<5000000:
#             #print ('1',((- d_k3_destination)/(d_k3_destination_max))*(1), done)
#             #print('2', (0.00005)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),100))
            
#             if done3 or k3_done_label==1:
#                 reward_k3 = 1
#                 # k3_done_label = 1
#             if done4 or k4_done_label==1:
#                 reward_k4 = 1
#             # print((0.00005)*min((R_k3_SN1+R_k3_SN2+R_k3_SN3),50),reward_k2,reward_k3, reward_k4)
#             # if done:
#             #     if label_done==0:
#             #         observation_k3_done = observation_k1
#             #         action_k1_done = action_k1
#             #         reward_k1_done = reward_k1
#             #         observation_k1_done_ = observation_k1_
#             #         h_count_done = h_count
#             #         label_done = 1
#             # if label_done==1:
#             #         observation_k1 = observation_k1_done
#             #         action_k1 = action_k1_done
#             #         reward_k1 = reward_k1_done
#             #         observation_k1_ =observation_k1_done_
#             # print((d_k1_destination_max-d_k1_destination)/(d_k1_destination_max)*(2.0))   
#             # print(reward_k1, label_done)
            
#             # RL learn from this transition
#             x_5 = observation_k5%xy_squre
#             y_5 = int(observation_k5/xy_squre)
#             # [x_SN1, y_SN1] = [0, 15]
#             # [x_SN2, y_SN2] = [5, 17]
#             # [x_SN3, y_SN3] = [13, 17]
#             # [x_SN4, y_SN4] = [3, 1]
#             # [x_SN5, y_SN5] = [11, 3]
#             # [x_SN6, y_SN6] = [15, 17]
            
#             d_k5_SN1 = (math.sqrt((x_5-x_SN1)*(x_5-x_SN1)+(y_5-y_SN1)*(y_5-y_SN1)))*Delta_xy+1
#             d_k5_SN2 = (math.sqrt((x_5-x_SN2)*(x_5-x_SN2)+(y_5-y_SN2)*(y_5-y_SN2)))*Delta_xy+1
#             d_k5_SN3 = (math.sqrt((x_5-x_SN3)*(x_5-x_SN3)+(y_5-y_SN3)*(y_5-y_SN3)))*Delta_xy+1
#             d_k5_SN4 = (math.sqrt((x_5-x_SN4-xy_squre)*(x_5-x_SN4-xy_squre)+(y_5-y_SN4)*(y_5-y_SN4)))*Delta_xy+1
#             d_k5_SN5 = (math.sqrt((x_5-x_SN5-xy_squre)*(x_5-x_SN5-xy_squre)+(y_5-y_SN5)*(y_5-y_SN5)))*Delta_xy+1
#             d_k5_SN6 = (math.sqrt((x_5-x_SN6-xy_squre)*(x_5-x_SN6-xy_squre)+(y_5-y_SN6)*(y_5-y_SN6)))*Delta_xy+1  
#             d_k5_destination = (math.sqrt((x_5-x_k5_destination)*(x_5-x_k5_destination)+(y_5-y_k5_destination)*(y_5-y_k5_destination)))*Delta_xy
#             beta_k5_SN1 = math.pow(10, -3)*math.pow(d_k5_SN1, 2.2)
#             beta_k5_SN2 = math.pow(10, -3)*math.pow(d_k5_SN2, 2.2)
#             beta_k5_SN3 = math.pow(10, -3)*math.pow(d_k5_SN3, 2.2) 
#             beta_k5_SN4 = math.pow(10, -3)*math.pow(d_k5_SN4, 2.2)
#             beta_k5_SN5 = math.pow(10, -3)*math.pow(d_k5_SN5, 2.2)
#             beta_k5_SN6 = math.pow(10, -3)*math.pow(d_k5_SN6, 2.2) 
#             theta_k5_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_5)/d_k5_SN1))
#             theta_k5_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_5)/d_k5_SN2))
#             theta_k5_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_5)/d_k5_SN3))
#             theta_k5_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4+xy_squre-x_5)/d_k5_SN4))
#             theta_k5_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5+xy_squre-x_5)/d_k5_SN5))
#             theta_k5_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6+xy_squre-x_5)/d_k5_SN6))
#             g_k5_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN4), math.exp(1)**(-1*1j*theta_k5_SN4*2)])
#             g_k5_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN5), math.exp(1)**(-1*1j*theta_k5_SN5*2)])
#             g_k5_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN6), math.exp(1)**(-1*1j*theta_k5_SN6*2)])
#             g_k5_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN1), math.exp(1)**(-1*1j*theta_k5_SN1*2)])
#             g_k5_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN2), math.exp(1)**(-1*1j*theta_k5_SN2*2)])
#             g_k5_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN3), math.exp(1)**(-1*1j*theta_k5_SN3*2)])
#             g_k5_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN1 = g_k5_SN1_LOS + g_k5_SN1_NLOS
#             g_k5_SN2 = g_k5_SN2_LOS + g_k5_SN2_NLOS
#             g_k5_SN3 = g_k5_SN3_LOS + g_k5_SN3_NLOS
#             g_k5_SN4 = g_k5_SN4_LOS + g_k5_SN4_NLOS
#             g_k5_SN5 = g_k5_SN5_LOS + g_k5_SN5_NLOS
#             g_k5_SN6 = g_k5_SN6_LOS + g_k5_SN6_NLOS
#             h_k5_SN1 = math.sqrt(beta_k5_SN1)*g_k5_SN1
#             h_k5_SN2 = math.sqrt(beta_k5_SN2)*g_k5_SN2
#             h_k5_SN3 = math.sqrt(beta_k5_SN3)*g_k5_SN3
#             h_k5_SN4 = math.sqrt(beta_k5_SN4)*g_k5_SN4            
#             h_k5_SN5 = math.sqrt(beta_k5_SN5)*g_k5_SN5
#             h_k5_SN6 = math.sqrt(beta_k5_SN6)*g_k5_SN6
#             #communication model
#             d_k5_SN7 = (math.sqrt((x_5-x_SN7-xy_squre)*(x_5-x_SN7-xy_squre)+(y_5-y_SN7-xy_squre)*(y_5-y_SN7-xy_squre)))*Delta_xy+1
#             d_k5_SN8 = (math.sqrt((x_5-x_SN8-xy_squre)*(x_5-x_SN8-xy_squre)+(y_5-y_SN8-xy_squre)*(y_5-y_SN8-xy_squre)))*Delta_xy+1
#             d_k5_SN9 = (math.sqrt((x_5-x_SN9-xy_squre)*(x_5-x_SN9-xy_squre)+(y_5-y_SN9-xy_squre)*(y_5-y_SN9-xy_squre)))*Delta_xy+1
#             beta_k5_SN7 = math.pow(10, -3)*math.pow(d_k5_SN7, 2.2)
#             beta_k5_SN8 = math.pow(10, -3)*math.pow(d_k5_SN8, 2.2)
#             beta_k5_SN9 = math.pow(10, -3)*math.pow(d_k5_SN9, 2.2) 
#             theta_k5_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_5)/d_k5_SN7))
#             theta_k5_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_5)/d_k5_SN8))
#             theta_k5_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_5)/d_k5_SN9))
#             g_k5_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN7), math.exp(1)**(-1*1j*theta_k5_SN7*2)])
#             g_k5_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN8), math.exp(1)**(-1*1j*theta_k5_SN8*2)])
#             g_k5_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k5_SN9), math.exp(1)**(-1*1j*theta_k5_SN9*2)])
#             g_k5_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k5_SN7 = g_k5_SN7_LOS + g_k5_SN7_NLOS
#             g_k5_SN8 = g_k5_SN8_LOS + g_k5_SN8_NLOS
#             g_k5_SN9 = g_k5_SN9_LOS + g_k5_SN9_NLOS
#             h_k5_SN7 = math.sqrt(beta_k5_SN7)*g_k5_SN7            
#             h_k5_SN8 = math.sqrt(beta_k5_SN8)*g_k5_SN8
#             h_k5_SN9 = math.sqrt(beta_k5_SN9)*g_k5_SN9
#             H_k5_PART3 = np.array([h_k5_SN7[0],h_k5_SN7[1],h_k5_SN7[2],h_k5_SN8[0],h_k5_SN8[1],h_k5_SN8[2],h_k5_SN9[0],h_k5_SN9[1],h_k5_SN9[2]]).T
#             H_k5_PART1 = np.array([h_k5_SN1[0],h_k5_SN1[1],h_k5_SN1[2],h_k5_SN2[0],h_k5_SN2[1],h_k5_SN2[2],h_k5_SN3[0],h_k5_SN3[1],h_k5_SN3[2]]).T
#             H_k5_PART2 = np.array([h_k5_SN4[0],h_k5_SN4[1],h_k5_SN4[2],h_k5_SN5[0],h_k5_SN5[1],h_k5_SN5[2],h_k5_SN6[0],h_k5_SN6[1],h_k5_SN6[2]]).T

    
    
#             #communication model
#             H_k5 = np.array([H_k5_PART1, H_k5_PART2, H_k5_PART3]).T
#             W_bar_k5 = H_k5.dot(lg.inv(H_k5.conj().T.dot(H_k5)))
#             w_k5_k5 = W_bar_k5[:, 0]/(np.linalg.norm(W_bar_k5[:, 0]))
#             # w_k5_k6 = W_bar_k5[:, 1]/(np.linalg.norm(W_bar_k5[:, 1]))
#             # w_k5_SN6=w_k5_k6[4:6]
#             # w_k5_SN5=w_k5_k6[2:4]
#             # w_k5_SN4=w_k5_k6[0:2]
#             w_k5_SN3=w_k5_k5[6:9]
#             w_k5_SN2=w_k5_k5[3:6]
#             w_k5_SN1=w_k5_k5[0:3]
#             # w_check3 = np.linalg.norm(w_k3_SN1)
#             # w_check4 = np.linalg.norm(w_k3_SN2)
#             # w_check3 = np.linalg.norm(w_k3_SN3)
#             order_k5_SN1 = w_k5_SN1.conj().T.dot(h_k5_SN1)
#             order_k5_SN2 = w_k5_SN2.conj().T.dot(h_k5_SN2)
#             order_k5_SN3 = w_k5_SN3.conj().T.dot(h_k5_SN3)
#             order_k5_SN1_positive = np.linalg.norm(order_k5_SN1)
#             order_k5_SN2_positive = np.linalg.norm(order_k5_SN2)
#             order_k5_SN3_positive = np.linalg.norm(order_k5_SN3)
#             # print(order_k3_SN1_positive, order_k3_SN2_positive, order_k3_SN3_positive)
    
#             x_6 = observation_k6%xy_squre
#             y_6 = int(observation_k6/xy_squre)
#             d_k6_SN4 = (math.sqrt((x_6-x_SN4)*(x_6-x_SN4)+(y_6-y_SN4)*(y_6-y_SN4)))*Delta_xy+1
#             d_k6_SN5 = (math.sqrt((x_6-x_SN5)*(x_6-x_SN5)+(y_6-y_SN5)*(y_6-y_SN5)))*Delta_xy+1
#             d_k6_SN6 = (math.sqrt((x_6-x_SN6)*(x_6-x_SN6)+(y_6-y_SN6)*(y_6-y_SN6)))*Delta_xy+1
#             d_k6_SN1 = (math.sqrt((x_6+xy_squre-x_SN1)*(x_6+xy_squre-x_SN1)+(y_6-y_SN1)*(y_6-y_SN1)))*Delta_xy+1
#             d_k6_SN2 = (math.sqrt((x_6+xy_squre-x_SN2)*(x_6+xy_squre-x_SN2)+(y_6-y_SN2)*(y_6-y_SN2)))*Delta_xy+1
#             d_k6_SN3 = (math.sqrt((x_6+xy_squre-x_SN3)*(x_6+xy_squre-x_SN3)+(y_6-y_SN3)*(y_6-y_SN3)))*Delta_xy+1
#             d_k6_destination = (math.sqrt((x_6-x_k6_destination)*(x_6-x_k6_destination)+(y_6-y_k6_destination)*(y_6-y_k6_destination)))*Delta_xy
#             beta_k6_SN4 = math.pow(10, -3)*math.pow(d_k6_SN4, 2.2)
#             beta_k6_SN5 = math.pow(10, -3)*math.pow(d_k6_SN5, 2.2)
#             beta_k6_SN6 = math.pow(10, -3)*math.pow(d_k6_SN6, 2.2)
#             beta_k6_SN1 = math.pow(10, -3)*math.pow(d_k6_SN1, 2.2)
#             beta_k6_SN2 = math.pow(10, -3)*math.pow(d_k6_SN2, 2.2)
#             beta_k6_SN3 = math.pow(10, -3)*math.pow(d_k6_SN3, 2.2)
#             theta_k6_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_6)/d_k6_SN4))
#             theta_k6_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_6)/d_k6_SN5))
#             theta_k6_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_6)/d_k6_SN6))
#             theta_k6_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_6-xy_squre)/d_k6_SN1))
#             theta_k6_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_6-xy_squre)/d_k6_SN2))
#             theta_k6_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_6-xy_squre)/d_k6_SN3))
#             g_k6_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN1), math.exp(1)**(-1*1j*theta_k6_SN1*2)])
#             g_k6_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN2), math.exp(1)**(-1*1j*theta_k6_SN2*2)])
#             g_k6_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN3), math.exp(1)**(-1*1j*theta_k6_SN3*2)])
#             g_k6_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN4), math.exp(1)**(-1*1j*theta_k6_SN4*2)])
#             g_k6_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN5), math.exp(1)**(-1*1j*theta_k6_SN5*2)])
#             g_k6_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN6), math.exp(1)**(-1*1j*theta_k6_SN6*2)])

#             g_k6_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN4 = g_k6_SN4_LOS + g_k6_SN4_NLOS
#             g_k6_SN5 = g_k6_SN5_LOS + g_k6_SN5_NLOS
#             g_k6_SN6 = g_k6_SN6_LOS + g_k6_SN6_NLOS
#             g_k6_SN1 = g_k6_SN1_LOS + g_k6_SN1_NLOS
#             g_k6_SN2 = g_k6_SN2_LOS + g_k6_SN2_NLOS
#             g_k6_SN3 = g_k6_SN3_LOS + g_k6_SN3_NLOS
#             h_k6_SN4 = math.sqrt(beta_k6_SN4)*g_k6_SN4            
#             h_k6_SN5 = math.sqrt(beta_k6_SN5)*g_k6_SN5
#             h_k6_SN6 = math.sqrt(beta_k6_SN6)*g_k6_SN6
#             h_k6_SN1 = math.sqrt(beta_k6_SN1)*g_k6_SN1            
#             h_k6_SN2 = math.sqrt(beta_k6_SN2)*g_k6_SN2
#             h_k6_SN3 = math.sqrt(beta_k6_SN3)*g_k6_SN3
#             #communication model
#             d_k6_SN7 = (math.sqrt((x_6-x_SN7)*(x_6-x_SN7)+(y_6-y_SN7-xy_squre)*(y_6-y_SN7-xy_squre)))*Delta_xy+1
#             d_k6_SN8 = (math.sqrt((x_6-x_SN8)*(x_6-x_SN8)+(y_6-y_SN8-xy_squre)*(y_6-y_SN8-xy_squre)))*Delta_xy+1
#             d_k6_SN9 = (math.sqrt((x_6-x_SN9)*(x_6-x_SN9)+(y_6-y_SN9-xy_squre)*(y_6-y_SN9-xy_squre)))*Delta_xy+1
#             beta_k6_SN7 = math.pow(10, -3)*math.pow(d_k6_SN7, 2.2)
#             beta_k6_SN8 = math.pow(10, -3)*math.pow(d_k6_SN8, 2.2)
#             beta_k6_SN9 = math.pow(10, -3)*math.pow(d_k6_SN9, 2.2) 
#             theta_k6_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_6)/d_k6_SN7))
#             theta_k6_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_6)/d_k6_SN8))
#             theta_k6_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_6)/d_k6_SN9))
#             g_k6_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN7), math.exp(1)**(-1*1j*theta_k6_SN7*2)])
#             g_k6_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN8), math.exp(1)**(-1*1j*theta_k6_SN8*2)])
#             g_k6_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k6_SN9), math.exp(1)**(-1*1j*theta_k6_SN9*2)])
#             g_k6_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k6_SN7 = g_k6_SN7_LOS + g_k6_SN7_NLOS
#             g_k6_SN8 = g_k6_SN8_LOS + g_k6_SN8_NLOS
#             g_k6_SN9 = g_k6_SN9_LOS + g_k6_SN9_NLOS
#             h_k6_SN7 = math.sqrt(beta_k6_SN7)*g_k6_SN7            
#             h_k6_SN8 = math.sqrt(beta_k6_SN8)*g_k6_SN8
#             h_k6_SN9 = math.sqrt(beta_k6_SN9)*g_k6_SN9
#             H_k6_PART3 = np.array([h_k6_SN7[0],h_k6_SN7[1],h_k6_SN7[2],h_k6_SN8[0],h_k6_SN8[1],h_k6_SN8[2],h_k6_SN9[0],h_k6_SN9[1],h_k6_SN9[2]]).T
#             H_k6_PART1 = np.array([h_k6_SN1[0],h_k6_SN1[1],h_k6_SN1[2],h_k6_SN2[0],h_k6_SN2[1],h_k6_SN2[2],h_k6_SN3[0],h_k6_SN3[1],h_k6_SN3[2]]).T
#             H_k6_PART2 = np.array([h_k6_SN4[0],h_k6_SN4[1],h_k6_SN4[2],h_k6_SN5[0],h_k6_SN5[1],h_k6_SN5[2],h_k6_SN6[0],h_k6_SN6[1],h_k6_SN6[2]]).T

            
#             #communication model
#             H_k6 = np.array([H_k6_PART1, H_k6_PART2, H_k6_PART3]).T
#             W_bar_k6 = H_k6.dot(lg.inv(H_k6.conj().T.dot(H_k6)))
#             # w_k6_k5 = W_bar_k6[:, 0]/(np.linalg.norm(W_bar_k6[:, 0]))
#             w_k6_k6 = W_bar_k6[:, 1]/(np.linalg.norm(W_bar_k6[:, 1]))
#             w_k6_SN6=w_k6_k6[6:9]
#             w_k6_SN5=w_k6_k6[3:6]
#             w_k6_SN4=w_k6_k6[0:3]
#             # w_k6_SN3=w_k6_k5[4:6]
#             # w_k6_SN2=w_k6_k5[2:4]
#             # w_k6_SN1=w_k6_k5[0:2]
#             # w_check3 = np.linalg.norm(w_k3_SN4)
#             # w_check4 = np.linalg.norm(w_k3_SN5)
#             # w_check3 = np.linalg.norm(w_k3_SN6)
#             order_k6_SN4 = w_k6_SN4.conj().T.dot(h_k6_SN4)
#             order_k6_SN5 = w_k6_SN5.conj().T.dot(h_k6_SN5)
#             order_k6_SN6 = w_k6_SN6.conj().T.dot(h_k6_SN6)
#             order_k6_SN4_positive = np.linalg.norm(order_k6_SN4)
#             order_k6_SN5_positive = np.linalg.norm(order_k6_SN5)
#             order_k6_SN6_positive = np.linalg.norm(order_k6_SN6)
#             # print(order_k3_SN1_positive, order_k3_SN5_positive, order_k3_SN3_positive)
    
#             x_9 = observation_k9%xy_squre
#             y_9 = int(observation_k9/xy_squre)
#             d_k9_SN4 = (math.sqrt((x_9-x_SN4)*(x_9-x_SN4)+(y_9+xy_squre-y_SN4)*(y_9+xy_squre-y_SN4)))*Delta_xy+1
#             d_k9_SN5 = (math.sqrt((x_9-x_SN5)*(x_9-x_SN5)+(y_9+xy_squre-y_SN5)*(y_9+xy_squre-y_SN5)))*Delta_xy+1
#             d_k9_SN6 = (math.sqrt((x_9-x_SN6)*(x_9-x_SN6)+(y_9+xy_squre-y_SN6)*(y_9+xy_squre-y_SN6)))*Delta_xy+1
#             d_k9_SN1 = (math.sqrt((x_9+xy_squre-x_SN1)*(x_9+xy_squre-x_SN1)+(y_9+xy_squre-y_SN1)*(y_9+xy_squre-y_SN1)))*Delta_xy+1
#             d_k9_SN2 = (math.sqrt((x_9+xy_squre-x_SN2)*(x_9+xy_squre-x_SN2)+(y_9+xy_squre-y_SN2)*(y_9+xy_squre-y_SN2)))*Delta_xy+1
#             d_k9_SN3 = (math.sqrt((x_9+xy_squre-x_SN3)*(x_9+xy_squre-x_SN3)+(y_9+xy_squre-y_SN3)*(y_9+xy_squre-y_SN3)))*Delta_xy+1
#             d_k9_destination = (math.sqrt((x_9-x_k9_destination)*(x_9-x_k9_destination)+(y_9-y_k9_destination)*(y_9-y_k9_destination)))*Delta_xy
#             beta_k9_SN4 = math.pow(10, -3)*math.pow(d_k9_SN4, 2.2)
#             beta_k9_SN5 = math.pow(10, -3)*math.pow(d_k9_SN5, 2.2)
#             beta_k9_SN6 = math.pow(10, -3)*math.pow(d_k9_SN6, 2.2)
#             beta_k9_SN1 = math.pow(10, -3)*math.pow(d_k9_SN1, 2.2)
#             beta_k9_SN2 = math.pow(10, -3)*math.pow(d_k9_SN2, 2.2)
#             beta_k9_SN3 = math.pow(10, -3)*math.pow(d_k9_SN3, 2.2)
#             theta_k9_SN4 = math.exp(1)**(-1*1j*(math.pi*(x_SN4-x_9)/d_k9_SN4))
#             theta_k9_SN5 = math.exp(1)**(-1*1j*(math.pi*(x_SN5-x_9)/d_k9_SN5))
#             theta_k9_SN6 = math.exp(1)**(-1*1j*(math.pi*(x_SN6-x_9)/d_k9_SN6))
#             theta_k9_SN1 = math.exp(1)**(-1*1j*(math.pi*(x_SN1-x_9-xy_squre)/d_k9_SN1))
#             theta_k9_SN2 = math.exp(1)**(-1*1j*(math.pi*(x_SN2-x_9-xy_squre)/d_k9_SN2))
#             theta_k9_SN3 = math.exp(1)**(-1*1j*(math.pi*(x_SN3-x_9-xy_squre)/d_k9_SN3))
#             g_k9_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN4), math.exp(1)**(-1*1j*theta_k9_SN4*2)])
#             g_k9_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN5), math.exp(1)**(-1*1j*theta_k9_SN5*2)])
#             g_k9_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN6), math.exp(1)**(-1*1j*theta_k9_SN6*2)])
#             g_k9_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN1), math.exp(1)**(-1*1j*theta_k9_SN1*2)])
#             g_k9_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN2), math.exp(1)**(-1*1j*theta_k9_SN2*2)])
#             g_k9_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN3), math.exp(1)**(-1*1j*theta_k9_SN3*2)])
#             g_k9_SN4_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN5_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN6_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN1_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN2_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN3_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN4 = g_k9_SN4_LOS + g_k9_SN4_NLOS
#             g_k9_SN5 = g_k9_SN5_LOS + g_k9_SN5_NLOS
#             g_k9_SN6 = g_k9_SN6_LOS + g_k9_SN6_NLOS
#             g_k9_SN1 = g_k9_SN1_LOS + g_k9_SN1_NLOS
#             g_k9_SN2 = g_k9_SN2_LOS + g_k9_SN2_NLOS
#             g_k9_SN3 = g_k9_SN3_LOS + g_k9_SN3_NLOS
#             h_k9_SN4 = math.sqrt(beta_k9_SN4)*g_k9_SN4            
#             h_k9_SN5 = math.sqrt(beta_k9_SN5)*g_k9_SN5
#             h_k9_SN6 = math.sqrt(beta_k9_SN6)*g_k9_SN6
#             h_k9_SN1 = math.sqrt(beta_k9_SN1)*g_k9_SN1            
#             h_k9_SN2 = math.sqrt(beta_k9_SN2)*g_k9_SN2
#             h_k9_SN3 = math.sqrt(beta_k9_SN3)*g_k9_SN3
            
#             d_k9_SN7 = (math.sqrt((x_9-x_SN7)*(x_9-x_SN7)+(y_9-y_SN7)*(y_9-y_SN7)))*Delta_xy+1
#             d_k9_SN8 = (math.sqrt((x_9-x_SN8)*(x_9-x_SN8)+(y_9-y_SN8)*(y_9-y_SN8)))*Delta_xy+1
#             d_k9_SN9 = (math.sqrt((x_9-x_SN9)*(x_9-x_SN9)+(y_9-y_SN9)*(y_9-y_SN9)))*Delta_xy+1
#             beta_k9_SN7 = math.pow(10, -3)*math.pow(d_k9_SN7, 2.2)
#             beta_k9_SN8 = math.pow(10, -3)*math.pow(d_k9_SN8, 2.2)
#             beta_k9_SN9 = math.pow(10, -3)*math.pow(d_k9_SN9, 2.2) 
#             theta_k9_SN7 = math.exp(1)**(-1*1j*(math.pi*(x_SN7+xy_squre-x_9)/d_k9_SN7))
#             theta_k9_SN8 = math.exp(1)**(-1*1j*(math.pi*(x_SN8+xy_squre-x_9)/d_k9_SN8))
#             theta_k9_SN9 = math.exp(1)**(-1*1j*(math.pi*(x_SN9+xy_squre-x_9)/d_k9_SN9))
#             g_k9_SN7_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN7), math.exp(1)**(-1*1j*theta_k9_SN7*2)])
#             g_k9_SN8_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN8), math.exp(1)**(-1*1j*theta_k9_SN8*2)])
#             g_k9_SN9_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k9_SN9), math.exp(1)**(-1*1j*theta_k9_SN9*2)])
#             g_k9_SN7_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN8_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN9_NLOS = math.sqrt(1/2)*np.random.rayleigh(size=(Antenna_L))
#             g_k9_SN7 = g_k9_SN7_LOS + g_k9_SN7_NLOS
#             g_k9_SN8 = g_k9_SN8_LOS + g_k9_SN8_NLOS
#             g_k9_SN9 = g_k9_SN9_LOS + g_k9_SN9_NLOS
#             h_k9_SN7 = math.sqrt(beta_k9_SN7)*g_k9_SN7            
#             h_k9_SN8 = math.sqrt(beta_k9_SN8)*g_k9_SN8
#             h_k9_SN9 = math.sqrt(beta_k9_SN9)*g_k9_SN9
#             H_k9_PART3 = np.array([h_k9_SN7[0],h_k9_SN7[1],h_k9_SN7[2],h_k9_SN8[0],h_k9_SN8[1],h_k9_SN8[2],h_k9_SN9[0],h_k9_SN9[1],h_k9_SN9[2]]).T
#             H_k9_PART1 = np.array([h_k9_SN1[0],h_k9_SN1[1],h_k9_SN1[2],h_k9_SN2[0],h_k9_SN2[1],h_k9_SN2[2],h_k9_SN3[0],h_k9_SN3[1],h_k9_SN3[2]]).T
#             H_k9_PART2 = np.array([h_k9_SN4[0],h_k9_SN4[1],h_k9_SN4[2],h_k9_SN5[0],h_k9_SN5[1],h_k9_SN5[2],h_k9_SN6[0],h_k9_SN6[1],h_k9_SN6[2]]).T

            
#             #communication model
#             H_k9 = np.array([H_k9_PART1, H_k9_PART2, H_k9_PART3]).T
#             # try:
#             #     W_bar_k9 = H_k9.dot(lg.inv(H_k9.conj().T.dot(H_k9)))
#             # except:
#             #     W_bar_k9 = W_bar_k9
#             # else:
#             #     W_bar_k9 = H_k9.dot(lg.inv(H_k9.conj().T.dot(H_k9)))
#             W_bar_k9 = H_k9.dot(lg.inv(H_k9.conj().T.dot(H_k9)))
#             # w_k9_k1 = W_bar_k9[:, 0]/(np.linalg.norm(W_bar_k9[:, 0]))
#             w_k9_k9 = W_bar_k9[:, 2]/(np.linalg.norm(W_bar_k9[:, 2]))
#             # w_k9_SN6 = W_bar_k9[:, 2]/(np.linalg.norm(W_bar_k9[:, 2]))
#             # w_k9_SN1 = W_bar_k9[:, 3]/(np.linalg.norm(W_bar_k9[:, 3]))
#             # w_k9_SN2 = W_bar_k9[:, 4]/(np.linalg.norm(W_bar_k9[:, 4]))
#             # w_k9_SN3 = W_bar_k9[:, 5]/(np.linalg.norm(W_bar_k9[:, 5]))
#             # w_check1 = np.linalg.norm(w_k1_SN4)
#             # w_check9 = np.linalg.norm(w_k1_SN5)
#             # w_check3 = np.linalg.norm(w_k1_SN6)
#             w_k9_SN9=w_k9_k9[6:9]
#             w_k9_SN8=w_k9_k9[3:6]
#             w_k9_SN7=w_k9_k9[0:3]
#             # w_k9_SN3=w_k9_k1[4:6]
#             # w_k9_SN2=w_k9_k1[2:4]
#             # w_k9_SN1=w_k9_k1[0:2]
#             # w_check1 = np.linalg.norm(w_k1_SN4)
#             # w_check9 = np.linalg.norm(w_k1_SN5)
#             # w_check3 = np.linalg.norm(w_k1_SN6)
#             order_k9_SN7 = w_k9_SN7.conj().T.dot(h_k9_SN7)
#             order_k9_SN8 = w_k9_SN8.conj().T.dot(h_k9_SN8)
#             order_k9_SN9 = w_k9_SN9.conj().T.dot(h_k9_SN9)
#             order_k9_SN7_positive = np.linalg.norm(order_k9_SN7)
#             order_k9_SN8_positive = np.linalg.norm(order_k9_SN8)
#             order_k9_SN9_positive = np.linalg.norm(order_k9_SN9)
    
#             # observation_k32 = observation_k3 + observation_k4*xy_squre*xy_squre
#             # RL choose action based on observation_k3
#             if k5_done_label == 0:
#                 action_k5 = RL5.choose_action(observation_k5)
#             if k6_done_label == 0:
#                 action_k6 = RL6.choose_action(observation_k6)
#             # action_k3 = action_k32%ACTION_SHAPE_single
#             # action_k4 = int(action_k32/ACTION_SHAPE_single)
#             order_k5 = [order_k5_SN1_positive, order_k5_SN2_positive, order_k5_SN3_positive]
#             order_k5_index = [order_k5[0] for order_k5 in sorted(enumerate(order_k5),key=lambda i:i[1], reverse=True)]
#             p_k5_SN_index = M_k
#             for i in order_k5_index:
#                 exec('''p_k5_SN{}_int = p_c63[int(action_k5/5), p_k5_SN_index-1]
# p_k5_SN{} = p_k5_SN{}_int*P_SN_mav/(6+1)
# p_k5_SN_index = p_k5_SN_index-1'''.format(i+1, i+1, i+1))
            
#             order_k5_index = [order_k5[0] for order_k5 in sorted(enumerate(order_k5),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k5_SN{} = (order_k5_SN{}_positive**2*p_k5_SN{})/(10**(-9))
# SINR_k5_SN{} = (order_k5_SN{}_positive**2*p_k5_SN{})/((np.linalg.norm(w_k5_SN{}.conj().T.dot(h_k5_SN{})))**2*p_k5_SN{}+10**(-9))
# SINR_k5_SN{} = (order_k5_SN{}_positive**2*p_k5_SN{})/((np.linalg.norm(w_k5_SN{}.conj().T.dot(h_k5_SN{})))**2*p_k5_SN{}+(np.linalg.norm(w_k5_SN{}.conj().T.dot(h_k5_SN{})))**2*p_k5_SN{}+10**(-9))'''.format(order_k5_index[2]+1,order_k5_index[2]+1,order_k5_index[2]+1,order_k5_index[1]+1,order_k5_index[1]+1,order_k5_index[1]+1,order_k5_index[1]+1,order_k5_index[2]+1,order_k5_index[2]+1,order_k5_index[0]+1,order_k5_index[0]+1,order_k5_index[0]+1,order_k5_index[0]+1,order_k5_index[2]+1,order_k5_index[2]+1,order_k5_index[0]+1,order_k5_index[1]+1,order_k5_index[1]+1))
    
#             order_k6 = [order_k6_SN4_positive, order_k6_SN5_positive, order_k6_SN6_positive]
#             order_k6_index = [order_k6[0] for order_k6 in sorted(enumerate(order_k6),key=lambda i:i[1], reverse=True)]
#             p_k6_SN_index = M_k
#             for i in order_k6_index:
#                 exec('''p_k6_SN{}_int = p_c63[int(action_k6/5), p_k6_SN_index-1]
# p_k6_SN{} = p_k6_SN{}_int*P_SN_mav/(6+1)
# p_k6_SN_index = p_k6_SN_index-1'''.format(i+4, i+4, i+4))
            
#             order_k6_index = [order_k6[0] for order_k6 in sorted(enumerate(order_k6),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k6_SN{} = (order_k6_SN{}_positive**2*p_k6_SN{})/(10**(-9))
# SINR_k6_SN{} = (order_k6_SN{}_positive**2*p_k6_SN{})/((np.linalg.norm(w_k6_SN{}.conj().T.dot(h_k6_SN{})))**2*p_k6_SN{}+10**(-9))
# SINR_k6_SN{} = (order_k6_SN{}_positive**2*p_k6_SN{})/((np.linalg.norm(w_k6_SN{}.conj().T.dot(h_k6_SN{})))**2*p_k6_SN{}+(np.linalg.norm(w_k6_SN{}.conj().T.dot(h_k6_SN{})))**2*p_k6_SN{}+10**(-9))'''.format(order_k6_index[2]+4,order_k6_index[2]+4,order_k6_index[2]+4,order_k6_index[1]+4,order_k6_index[1]+4,order_k6_index[1]+4,order_k6_index[1]+4,order_k6_index[2]+4,order_k6_index[2]+4,order_k6_index[0]+4,order_k6_index[0]+4,order_k6_index[0]+4,order_k6_index[0]+4,order_k6_index[2]+4,order_k6_index[2]+4,order_k6_index[0]+4,order_k6_index[1]+4,order_k6_index[1]+4))
#             if k9_done_label == 0:
#                 action_k9 = RL9.choose_action(observation_k9)
#             # action_k1 = action_k12%ACTION_SHAPE_single
#             # action_k2 = int(action_k12/ACTION_SHAPE_single)
#             order_k9 = [order_k9_SN7_positive, order_k9_SN8_positive, order_k9_SN9_positive]
#             order_k9_index = [order_k9[0] for order_k9 in sorted(enumerate(order_k9),key=lambda i:i[1], reverse=True)]
#             p_k9_SN_index = M_k
#             for i in order_k9_index:
#                 exec('''p_k9_SN{}_int = p_c63[int(action_k9/5), p_k9_SN_index-1]
# p_k9_SN{} = p_k9_SN{}_int*P_SN_mav/(6+1)
# p_k9_SN_index = p_k9_SN_index-1'''.format(i+7, i+7, i+7))
            
#             order_k9_index = [order_k9[0] for order_k9 in sorted(enumerate(order_k9),key=lambda i:i[1], reverse=True)]
#             exec('''SINR_k9_SN{} = (order_k9_SN{}_positive**2*p_k9_SN{})/(10**(-9))
# SINR_k9_SN{} = (order_k9_SN{}_positive**2*p_k9_SN{})/((np.linalg.norm(w_k9_SN{}.conj().T.dot(h_k9_SN{})))**2*p_k9_SN{}+70**(-9))
# SINR_k9_SN{} = (order_k9_SN{}_positive**2*p_k9_SN{})/((np.linalg.norm(w_k9_SN{}.conj().T.dot(h_k9_SN{})))**2*p_k9_SN{}+(np.linalg.norm(w_k9_SN{}.conj().T.dot(h_k9_SN{})))**2*p_k9_SN{}+10**(-9))'''.format(order_k9_index[2]+7,order_k9_index[2]+7,order_k9_index[2]+7,order_k9_index[1]+7,order_k9_index[1]+7,order_k9_index[1]+7,order_k9_index[1]+7,order_k9_index[2]+7,order_k9_index[2]+7,order_k9_index[0]+7,order_k9_index[0]+7,order_k9_index[0]+7,order_k9_index[0]+7,order_k9_index[2]+7,order_k9_index[2]+7,order_k9_index[0]+7,order_k9_index[1]+7,order_k9_index[1]+7))
#             R_k9_SN7 = math.log((1+SINR_k9_SN7),2)
#             R_k9_SN8 = math.log((1+SINR_k9_SN8),2)
#             R_k9_SN9 = math.log((1+SINR_k9_SN9),2)
#             R_k5_SN1 = math.log((1+SINR_k5_SN1),2)
#             R_k5_SN2 = math.log((1+SINR_k5_SN2),2)
#             R_k5_SN3 = math.log((1+SINR_k5_SN3),2)
#             R_k6_SN4 = math.log((1+SINR_k6_SN4),2)
#             R_k6_SN5 = math.log((1+SINR_k6_SN5),2)
#             R_k6_SN6 = math.log((1+SINR_k6_SN6),2)
            
#             # RL take action and get next observation_k5_k5 and reward
#             observation_k5_, reward_k5, done5 = env5.step(action_k5)
#             observation_k6_, reward_k6, done6 = env6.step(action_k6)
#             #observation_k5_, reward, done = env.step(action)
#             observation_k9_, reward_k9, done9 = env9.step(action_k9)
#             reward_k9 = (0.00005)*min((R_k9_SN7+R_k9_SN8+R_k9_SN9),50)
#             if done9 or k9_done_label==1:
#                 reward_k9 = 1
#             # reward_k5_reach = 0
#             # if done:
#             #     reward_k5_reach = 1
                
#             # reward_k5 = ((0.0)*min((R_k5_SN1+R_k5_SN2+R_k5_SN3),100)*0.01+(1)*max(0, (d_k5_destination_max-d_k5_destination))/(d_k5_destination_max))*(0.2)+(1.5)*reward_k5_reach
#             reward_k5 = (0.00005)*min((R_k5_SN1+R_k5_SN2+R_k5_SN3),50)
#             reward_k6 = (0.00005)*min((R_k6_SN4+R_k6_SN5+R_k6_SN6),50)
            
#             if done5 or k5_done_label==1:
#                 reward_k5 = 1
#                 # k5_done_label = 1
#             if done6 or k6_done_label==1:
#                 reward_k6 = 1
                
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
                 ep_R_k1 = R_k1_SN1+R_k1_SN2+R_k1_SN3+ep_R_k1
            
                
                 # N_k3_hnsa[observation_k1, action_k1, h_count] = N_k3_hnsa[observation_k1, action_k1, h_count]+1
                 # lr_k3 = (h_count_done+1)/(h_count_done+N_k3_hnsa[observation_k1, action_k1, h_count])
                 # b_k3_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k3)/(N_k3_hnsa[observation_k1, action_k1, h_count]*C_K))
                 #  # print(reward_k1)
                 #  # RL learn from this transition
                 #  # print(lr_k1, b_k1_done)
                 # RL3.learn(observation_k1, action_k1, reward_k1, observation_k1_, lr_k3, b_k3_done, done1)
                
                 # N_k5_hnsa[observation_k1, action_k1, h_count] = N_k5_hnsa[observation_k1, action_k1, h_count]+1
                 # lr_k5 = (h_count_done+1)/(h_count_done+N_k5_hnsa[observation_k1, action_k1, h_count])
                 # b_k5_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k5)/(N_k5_hnsa[observation_k1, action_k1, h_count]*C_K))
                 #  # print(reward_k1)
                 #  # RL learn from this transition
                 #  # print(lr_k1, b_k1_done)
                 # RL5.learn(observation_k1, action_k1, reward_k1, observation_k1_, lr_k5, b_k5_done, done1)

            
            if k2_done_label == 0:
                 N_k2_hnsa[observation_k2, action_k2, h_count] = N_k2_hnsa[observation_k2, action_k2, h_count]+1
                 lr_k2 = (h_count_done+1)/(h_count_done+N_k2_hnsa[observation_k2, action_k2, h_count])
                 b_k2_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k2)/(N_k2_hnsa[observation_k2, action_k2, h_count]*C_K))
                 # print(reward_k1)
                 # RL learn from this transition
                 # print(lr_k1, b_k1_done)
                 RL2.learn(observation_k2, action_k2, reward_k2, observation_k2_, lr_k2, b_k2_done, done2)
                 ep_reward_k2 += reward_k2
                 observation_k2 = observation_k2_
                 h_count_k2_done = h_count
                 ep_R_k2 = R_k2_SN4+R_k2_SN5+R_k2_SN6+ep_R_k2
            
                
                 # N_k4_hnsa[observation_k2, action_k2, h_count] = N_k4_hnsa[observation_k2, action_k2, h_count]+1
                 # lr_k4 = (h_count_done+1)/(h_count_done+N_k4_hnsa[observation_k2, action_k2, h_count])
                 # b_k4_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k4)/(N_k4_hnsa[observation_k2, action_k2, h_count]*C_K))
                 #  # print(reward_k1)
                 #  # RL learn from this transition
                 #  # print(lr_k4, b_k4_done)
                 # RL4.learn(observation_k2, action_k2, reward_k2, observation_k2_, lr_k4, b_k4_done, done2)
                
                 # N_k6_hnsa[observation_k2, action_k2, h_count] = N_k6_hnsa[observation_k2, action_k2, h_count]+1
                 # lr_k6 = (h_count_done+1)/(h_count_done+N_k6_hnsa[observation_k2, action_k2, h_count])
                 # b_k6_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k6)/(N_k6_hnsa[observation_k2, action_k2, h_count]*C_K))
                 #  # print(reward_k1)
                 #  # RL learn from this transition
                 #  # print(lr_k6, b_k6_done)
                 # RL6.learn(observation_k2, action_k2, reward_k2, observation_k2_, lr_k6, b_k6_done, done2)
                 # ep_reward_k1 += reward_k1
                 # observation_k2 = observation_k2_
                 # h_count_k1_done = h_count
     
            if done1:
                 k1_done_label = 1
                 # k1_done_label = 1
            if done2:
                 k2_done_label = 1
             # swap observation_k1
             # observation_k1 = observation_k12_%(xy_squre*xy_squre)
             # observation_k2 = int(observation_k12_/(xy_squre*xy_squre))
             # d_k1_destination_ = d_k1_destination
            
            
             # d_k1_destination_ = d_k1_destination
            # h_count=h_count+1
            # if k3_done_label == 0:
            #  # observation_k12_= observation_k1_ + observation_k2_*xy_squre*xy_squre
            #      N_k3_hnsa[observation_k3, action_k3, h_count] = N_k3_hnsa[observation_k3, action_k3, h_count]+1
            #      lr_k3 = (h_count_done+1)/(h_count_done+N_k3_hnsa[observation_k3, action_k3, h_count])
            #      b_k3_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k3)/(N_k3_hnsa[observation_k3, action_k3, h_count]*C_K))
            #      # print(reward_k3)
            #      # RL learn from this transition
            #      # print(lr_k3, b_k3_done)
            #      RL3.learn(observation_k3, action_k3, reward_k3, observation_k3_, lr_k3, b_k3_done, done3)
            #      ep_reward_k3 += reward_k3
            #      observation_k3 = observation_k3_
            #      h_count_k3_done = h_count
            #      ep_R_k3 = R_k3_SN1+R_k3_SN2+R_k3_SN3+ep_R_k3
            
                
            #      # N_k1_hnsa[observation_k3, action_k3, h_count] = N_k1_hnsa[observation_k3, action_k3, h_count]+1
            #      # lr_k1 = (h_count_done+1)/(h_count_done+N_k1_hnsa[observation_k3, action_k3, h_count])
            #      # b_k1_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k1)/(N_k1_hnsa[observation_k3, action_k3, h_count]*C_K))
            #      #  # print(reward_k3)
            #      #  # RL learn from this transition
            #      #  # print(lr_k3, b_k3_done)
            #      # RL1.learn(observation_k3, action_k3, reward_k3, observation_k3_, lr_k1, b_k1_done, done3)
            #      #  # ep_reward_k1 += reward_k2
            #      #  # observation_k2 = observation_k2_
            #      #  # h_count_k2_done = h_count
            #      # N_k5_hnsa[observation_k3, action_k3, h_count] = N_k5_hnsa[observation_k3, action_k3, h_count]+1
            #      # lr_k5 = (h_count_done+1)/(h_count_done+N_k5_hnsa[observation_k3, action_k3, h_count])
            #      # b_k5_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k5)/(N_k5_hnsa[observation_k3, action_k3, h_count]*C_K))
            #      #  # print(reward_k3)
            #      #  # RL learn from this transition
            #      #  # print(lr_k3, b_k3_done)
            #      # RL5.learn(observation_k3, action_k3, reward_k3, observation_k3_, lr_k5, b_k5_done, done3)
            
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
            #      ep_R_k4 = R_k4_SN4+R_k4_SN5+R_k4_SN6+ep_R_k4
            
                
            #      # N_k2_hnsa[observation_k4, action_k4, h_count] = N_k2_hnsa[observation_k4, action_k4, h_count]+1
            #      # lr_k2 = (h_count_done+1)/(h_count_done+N_k2_hnsa[observation_k4, action_k4, h_count])
            #      # b_k2_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k2)/(N_k2_hnsa[observation_k4, action_k4, h_count]*C_K))
            #      #  # print(reward_k2)
            #      #  # RL learn from this transition
            #      #  # print(lr_k2, b_k2_done)
            #      # RL2.learn(observation_k4, action_k4, reward_k4, observation_k4_, lr_k2, b_k2_done, done4)
            #      #  # ep_reward_k1 += reward_k1
            #      #  # observation_k4 = observation_k4_
            #      #  # h_count_k1_done = h_count
            #      # N_k6_hnsa[observation_k4, action_k4, h_count] = N_k6_hnsa[observation_k4, action_k4, h_count]+1
            #      # lr_k6 = (h_count_done+1)/(h_count_done+N_k6_hnsa[observation_k4, action_k4, h_count])
            #      # b_k6_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k6)/(N_k6_hnsa[observation_k4, action_k4, h_count]*C_K))
            #      #  # print(reward_k6)
            #      #  # RL learn from this transition
            #      #  # print(lr_k6, b_k6_done)
            #      # RL6.learn(observation_k4, action_k4, reward_k4, observation_k4_, lr_k6, b_k6_done, done4)
     
            # if done3:
            #      k3_done_label = 1
            #      # k1_done_label = 1
            # if done4:
            #      k4_done_label = 1
            #  # swap observation_k1
            #  # observation_k1 = observation_k12_%(xy_squre*xy_squre)
            #  # observation_k2 = int(observation_k12_/(xy_squre*xy_squre))
            #  # d_k1_destination_ = d_k1_destination
            
            
            #  # d_k1_destination_ = d_k1_destination
            # # h_count=h_count+1 
            # if k5_done_label == 0:
            # # observation_k12_= observation_k1_ + observation_k2_*xy_squre*xy_squre
            #     N_k5_hnsa[observation_k5, action_k5, h_count] = N_k5_hnsa[observation_k5, action_k5, h_count]+1
            #     lr_k5 = (h_count_done+1)/(h_count_done+N_k5_hnsa[observation_k5, action_k5, h_count])
            #     b_k5_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k5)/(N_k5_hnsa[observation_k5, action_k5, h_count]*C_K))
            #     RL5.learn(observation_k5, action_k5, reward_k5, observation_k5_, lr_k5, b_k5_done, done5)
            #     ep_reward_k5 += reward_k5
            #     observation_k5 = observation_k5_
            #     h_count_k5_done = h_count
            #     ep_R_k5 = R_k5_SN1+R_k5_SN2+R_k5_SN3+ep_R_k5
            
                
            #     # N_k1_hnsa[observation_k5, action_k5, h_count] = N_k1_hnsa[observation_k5, action_k5, h_count]+1
            #     # lr_k1 = (h_count_done+1)/(h_count_done+N_k1_hnsa[observation_k5, action_k5, h_count])
            #     # b_k1_done = 0.000001*math.sqrt(h_count_done**5*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k1)/(N_k1_hnsa[observation_k5, action_k5, h_count]*C_K))
            #     # # print(reward_k5)
            #     # # RL learn from this transition
            #     # # print(lr_k5, b_k5_done)
            #     # RL1.learn(observation_k5, action_k5, reward_k5, observation_k5_, lr_k1, b_k1_done, done5)
            #     # # ep_reward_k1 += reward_k2
            #     # # observation_k2 = observation_k2_
            #     # # h_count_k2_done = h_count
            #     # N_k3_hnsa[observation_k5, action_k5, h_count] = N_k3_hnsa[observation_k5, action_k5, h_count]+1
            #     # lr_k3 = (h_count_done+1)/(h_count_done+N_k3_hnsa[observation_k5, action_k5, h_count])
            #     # b_k3_done = 0.000001*math.sqrt(h_count_done**5*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k3)/(N_k3_hnsa[observation_k5, action_k5, h_count]*C_K))
            #     # RL3.learn(observation_k5, action_k5, reward_k5, observation_k5_, lr_k3, b_k3_done, done5)
            
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
            #     ep_R_k6 = R_k6_SN4+R_k6_SN5+R_k6_SN6+ep_R_k6
            
                
            #     # N_k2_hnsa[observation_k6, action_k6, h_count] = N_k2_hnsa[observation_k6, action_k6, h_count]+1
            #     # lr_k2 = (h_count_done+1)/(h_count_done+N_k2_hnsa[observation_k6, action_k6, h_count])
            #     # b_k2_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k2)/(N_k2_hnsa[observation_k6, action_k6, h_count]*C_K))
            #     # # print(reward_k2)
            #     # # RL learn from this transition
            #     # # print(lr_k2, b_k2_done)
            #     # RL2.learn(observation_k6, action_k6, reward_k6, observation_k6_, lr_k2, b_k2_done, done6)
                
            #     # N_k4_hnsa[observation_k6, action_k6, h_count] = N_k4_hnsa[observation_k6, action_k6, h_count]+1
            #     # lr_k4 = (h_count_done+1)/(h_count_done+N_k4_hnsa[observation_k6, action_k6, h_count])
            #     # b_k4_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k4)/(N_k4_hnsa[observation_k6, action_k6, h_count]*C_K))
            #     # # print(reward_k6)
            #     # # RL learn from this transition
            #     # # print(lr_k6, b_k6_done)
            #     # RL4.learn(observation_k6, action_k6, reward_k6, observation_k6_, lr_k4, b_k4_done, done6)
            
            if k7_done_label == 0:
                N_k7_hnsa[observation_k7, action_k7, h_count] = N_k7_hnsa[observation_k7, action_k7, h_count]+1
                lr_k7 = (h_count_done+1)/(h_count_done+N_k7_hnsa[observation_k7, action_k7, h_count])
                b_k7_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k7)/(N_k7_hnsa[observation_k7, action_k7, h_count]*C_K))
                # print(reward_k1)
                # RL learn from this transition
                # print(lr_k1, b_k1_done)
                RL7.learn(observation_k7, action_k7, reward_k7, observation_k7_, lr_k7, b_k7_done, done7)
                ep_reward_k7 += reward_k7
                observation_k7 = observation_k7_
                h_count_k7_done = h_count
                ep_R_k7 = R_k7_SN7+R_k7_SN8+R_k7_SN9+ep_R_k7
            
                
                # N_k8_hnsa[observation_k7, action_k7, h_count] = N_k8_hnsa[observation_k7, action_k7, h_count]+1
                # lr_k8 = (h_count_done+1)/(h_count_done+N_k8_hnsa[observation_k7, action_k7, h_count])
                # b_k8_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k8)/(N_k8_hnsa[observation_k7, action_k7, h_count]*C_K))
                # RL8.learn(observation_k7, action_k7, reward_k7, observation_k7_, lr_k8, b_k8_done, done7)
                
                # N_k9_hnsa[observation_k7, action_k7, h_count] = N_k9_hnsa[observation_k7, action_k7, h_count]+1
                # lr_k9 = (h_count_done+1)/(h_count_done+N_k9_hnsa[observation_k7, action_k7, h_count])
                # b_k9_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k9)/(N_k9_hnsa[observation_k7, action_k7, h_count]*C_K))
                # RL9.learn(observation_k7, action_k7, reward_k7, observation_k7_, lr_k9, b_k9_done, done7)
            if done7:
                k7_done_label = 1
                
                
            if k10_done_label == 0:
                N_k10_hnsa[observation_k10, action_k10, h_count] = N_k10_hnsa[observation_k10, action_k10, h_count]+1
                lr_k10 = (h_count_done+1)/(h_count_done+N_k10_hnsa[observation_k10, action_k10, h_count])
                b_k10_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k10)/(N_k10_hnsa[observation_k10, action_k10, h_count]*C_K))
                # print(reward_k1)
                # RL learn from this transition
                # print(lr_k1, b_k1_done)
                RL10.learn(observation_k10, action_k10, reward_k10, observation_k10_, lr_k10, b_k10_done, done10)
                ep_reward_k10 += reward_k10
                observation_k10 = observation_k10_
                h_count_k10_done = h_count
                ep_R_k10 = R_k10_SN10+R_k10_SN11+R_k10_SN12+ep_R_k10
                
            if done10:
                k10_done_label = 1
            # if k8_done_label == 0:
            #     N_k8_hnsa[observation_k8, action_k8, h_count] = N_k8_hnsa[observation_k8, action_k8, h_count]+1
            #     lr_k8 = (h_count_done+1)/(h_count_done+N_k8_hnsa[observation_k8, action_k8, h_count])
            #     b_k8_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k8)/(N_k8_hnsa[observation_k8, action_k8, h_count]*C_K))
            #     # print(reward_k1)
            #     # RL learn from this transition
            #     # print(lr_k1, b_k1_done)
            #     RL8.learn(observation_k8, action_k8, reward_k8, observation_k8_, lr_k8, b_k8_done, done8)
            #     ep_reward_k8 += reward_k8
            #     observation_k8 = observation_k8_
            #     h_count_k8_done = h_count
            #     ep_R_k8 = R_k8_SN7+R_k8_SN8+R_k8_SN9+ep_R_k8
            
                
            #     # N_k7_hnsa[observation_k8, action_k8, h_count] = N_k7_hnsa[observation_k8, action_k8, h_count]+1
            #     # lr_k7 = (h_count_done+1)/(h_count_done+N_k7_hnsa[observation_k8, action_k8, h_count])
            #     # b_k7_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k7)/(N_k7_hnsa[observation_k8, action_k8, h_count]*C_K))
            #     # RL7.learn(observation_k8, action_k8, reward_k8, observation_k8_, lr_k7, b_k7_done, done8)
                
            #     # N_k9_hnsa[observation_k8, action_k8, h_count] = N_k9_hnsa[observation_k8, action_k8, h_count]+1
            #     # lr_k9 = (h_count_done+1)/(h_count_done+N_k9_hnsa[observation_k8, action_k8, h_count])
            #     # b_k9_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k9)/(N_k9_hnsa[observation_k8, action_k8, h_count]*C_K))
            #     # RL9.learn(observation_k8, action_k8, reward_k8, observation_k8_, lr_k9, b_k9_done, done8)
            # if done8:
            #     k8_done_label = 1    
                
            # if k9_done_label == 0:
            #     N_k9_hnsa[observation_k9, action_k9, h_count] = N_k9_hnsa[observation_k9, action_k9, h_count]+1
            #     lr_k9 = (h_count_done+1)/(h_count_done+N_k9_hnsa[observation_k9, action_k9, h_count])
            #     b_k9_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k9)/(N_k9_hnsa[observation_k9, action_k9, h_count]*C_K))
            #     # print(reward_k1)
            #     # RL learn from this transition
            #     # print(lr_k1, b_k1_done)
            #     RL9.learn(observation_k9, action_k9, reward_k9, observation_k9_, lr_k9, b_k9_done, done9)
            #     ep_reward_k9 += reward_k9
            #     observation_k9 = observation_k9_
            #     h_count_k9_done = h_count
            #     ep_R_k9 = R_k9_SN7+R_k9_SN8+R_k9_SN9+ep_R_k9
                
            #     # N_k7_hnsa[observation_k9, action_k9, h_count] = N_k7_hnsa[observation_k9, action_k9, h_count]+1
            #     # lr_k7 = (h_count_done+1)/(h_count_done+N_k7_hnsa[observation_k9, action_k9, h_count])
            #     # b_k7_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k7)/(N_k7_hnsa[observation_k9, action_k9, h_count]*C_K))
            #     # RL7.learn(observation_k9, action_k9, reward_k9, observation_k9_, lr_k7, b_k7_done, done9)
                
            #     # N_k8_hnsa[observation_k9, action_k9, h_count] = N_k8_hnsa[observation_k9, action_k9, h_count]+1
            #     # lr_k8 = (h_count_done+1)/(h_count_done+N_k8_hnsa[observation_k9, action_k9, h_count])
            #     # b_k8_done = 0.000001*math.sqrt(h_count_done**3*math.log(xy_squre*xy_squre*ACTION_SHAPE*n_count_done*h_count_done/pho_k8)/(N_k8_hnsa[observation_k9, action_k9, h_count]*C_K))
            #     # RL8.learn(observation_k9, action_k9, reward_k9, observation_k9_, lr_k8, b_k8_done, done9)
            # if done9:
            #     k9_done_label = 1    
            
            # if done5:
            #     k5_done_label = 1
            #     # k1_done_label = 1
            # if done6:
            #     k6_done_label = 1
            # ep_reward_k12 += reward_k12
            # break while loop when end of this episode
            if k1_done_label == 1:
                if k2_done_label == 1:
                    # if k3_done_label == 1:
                    #     if k4_done_label == 1:
                    #         if k5_done_label == 1:
                    #             if k6_done_label == 1:
                                    if k7_done_label == 1:
                                        if k10_done_label == 1:
                                        #     if k9_done_label == 1:
                                                break
        # print('1',episode, h_count_k1_done+1, h_count_k2_done+1)
        # print('2',episode, h_count_k3_done+1, h_count_k4_done+1)
        # print('3',episode, h_count_k5_done+1, h_count_k6_done+1)
        # print('**********************************************************')##    print(episode, h_count, ep_reward_k2+ep_reward_k1+(h_count_done-ep_reward_k1)*reward_k1+(h_count_done-ep_reward_k2)*reward_k2)
        R_sum =(ep_R_k1/(h_count_k1_done+1)+ep_R_k2/(h_count_k2_done+1)+ep_R_k7/(h_count_k7_done+1)+ep_R_k10/(h_count_k10_done+1))/4
        reward_sum=(ep_reward_k10+ep_reward_k7+ep_reward_k1+ep_reward_k2)/4
        filename = 'ep_reward.txt'
        with open(filename,'a') as fileobject: #浣跨敤鈥榓'鏉ユ彁閱抪ython鐢ㄩ檮鍔犳ā寮忕殑鏂瑰紡鎵撳紑
              fileobject.write(str(reward_sum)+'\n')  
        # ep_sum2 = (ep_reward_k1+(h_count_done-h_count_k1_done-1)*1+ep_reward_k2+(h_count_done-h_count_k2_done-1)*1+ep_reward_k3+(h_count_done-h_count_k3_done-1)*1+ep_reward_k4+(h_count_done-h_count_k4_done-1)*1)/2
        filename = 'ep_R.txt'
        with open(filename,'a') as fileobject: #浣跨敤鈥榓'鏉ユ彁閱抪ython鐢ㄩ檮鍔犳ā寮忕殑鏂瑰紡鎵撳紑
              fileobject.write(str(R_sum)+'\n')  
        step_count=(h_count_k1_done+h_count_k2_done+h_count_k10_done+h_count_k7_done)/4
        filename = 'ep_step.txt'
        with open(filename,'a') as fileobject: #浣跨敤鈥榓'鏉ユ彁閱抪ython鐢ㄩ檮鍔犳ā寮忕殑鏂瑰紡鎵撳紑
              fileobject.write(str(step_count)+'\n')  
    # end of game
print('game over')
env.destroy()