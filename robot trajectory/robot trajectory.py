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
K = 1
M_k = 1
Antenna_L = 1
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
   
    
    N_k1_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    N_k2_hnsa = np.zeros([xy_squre*xy_squre, ACTION_SHAPE, h_count_done])
    
    
    #print(RL.q_table)
    # d_k1_destination_max = (math.sqrt((xy_squre)*(xy_squre)*2))*Delta_xy
    
    for episode in range(n_count_done):
        # initial observation_k1
        ep_R_k1 = 0
        ep_R_k2 = 0
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

        x_k1_array = []
        y_k1_array = []
        x_k2_array = []
        y_k2_array = []
       
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
            
            g_k1_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN1)])
            g_k1_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN2)])
            g_k1_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN3)])
            g_k1_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN4)])
            g_k1_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN5)])
            g_k1_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k1_SN6)])
            
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
            H_k1_PART1 = np.array([h_k1_SN1[0],h_k1_SN1[1],h_k1_SN2[0],h_k1_SN2[1],h_k1_SN3[0],h_k1_SN3[1]]).T
            H_k1_PART2 = np.array([h_k1_SN4[0],h_k1_SN4[1],h_k1_SN5[0],h_k1_SN5[1],h_k1_SN6[0],h_k1_SN6[1]]).T
    
                     
            #communication model
            H_k1 = np.array([H_k1_PART1, H_k1_PART2]).T
            
            W_bar_k1 = H_k1.dot(lg.inv(H_k1.conj().T.dot(H_k1)))
            w_k1_k1 = W_bar_k1[:, 0]/(np.linalg.norm(W_bar_k1[:, 0]))
           
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
            g_k2_SN4_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN4)])
            g_k2_SN5_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN5)])
            g_k2_SN6_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN6)])
            g_k2_SN1_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN1)])
            g_k2_SN2_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN2)])
            g_k2_SN3_LOS = math.sqrt(1/2)*np.array([1,math.exp(1)**(-1*1j*theta_k2_SN3)])
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
            
            H_k2_PART1 = np.array([h_k2_SN1[0],h_k2_SN1[1],h_k2_SN1[2],h_k2_SN1[3],h_k2_SN2[0],h_k2_SN2[1],h_k2_SN2[2],h_k2_SN2[3],h_k2_SN3[0],h_k2_SN3[1],h_k2_SN3[2],h_k2_SN3[3]]).T
            H_k2_PART2 = np.array([h_k2_SN4[0],h_k2_SN4[1],h_k2_SN4[2],h_k2_SN4[3],h_k2_SN5[0],h_k2_SN5[1],h_k2_SN5[2],h_k2_SN5[3],h_k2_SN6[0],h_k2_SN6[1],h_k2_SN6[2],h_k2_SN6[3]]).T
            
            #communication model
            H_k2 = np.array([H_k2_PART1, H_k2_PART2]).T
            
            W_bar_k2 = H_k2.dot(lg.inv(H_k2.conj().T.dot(H_k2)))
            # w_k2_k1 = W_bar_k2[:, 0]/(np.linalg.norm(W_bar_k2[:, 0]))
            w_k2_k2 = W_bar_k2[:, 1]/(np.linalg.norm(W_bar_k2[:, 1]))
           
            w_k2_SN6=w_k2_k2[4:6]
            w_k2_SN5=w_k2_k2[2:4]
            w_k2_SN4=w_k2_k2[0:2]
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
            
            R_k1_SN1 = math.log((1+SINR_k1_SN1),2)
            R_k1_SN2 = math.log((1+SINR_k1_SN2),2)
            R_k1_SN3 = math.log((1+SINR_k1_SN3),2)
            R_k2_SN4 = math.log((1+SINR_k2_SN4),2)
            R_k2_SN5 = math.log((1+SINR_k2_SN5),2)
            R_k2_SN6 = math.log((1+SINR_k2_SN6),2)  
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
            
                
                
     
            if done1:
                 k1_done_label = 1
                 # k1_done_label = 1
            if done2:
                 k2_done_label = 1
            
            # break while loop when end of this episode
            if k1_done_label == 1:
                if k2_done_label == 1:
                      break
        # print('1',episode, h_count_k1_done+1, h_count_k2_done+1)
        # print('2',episode, h_count_k3_done+1, h_count_k4_done+1)
        # print('3',episode, h_count_k5_done+1, h_count_k6_done+1)
        # print('**********************************************************')##    print(episode, h_count, ep_reward_k2+ep_reward_k1+(h_count_done-ep_reward_k1)*reward_k1+(h_count_done-ep_reward_k2)*reward_k2)
        for x in range(len(x_k1_array)):
            filename = 'x_k1'+str(episode)+"_"+str(h_count)+'.txt'
            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                  fileobject.write(str(x_k1_array[x])+'\n')
        for y in range(len(y_k1_array)):
            filename = 'y_k1'+str(episode)+"_"+str(h_count)+'.txt'
            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                  fileobject.write(str(y_k1_array[y])+'\n')
        for x in range(len(x_k2_array)):
            filename = 'x_k2'+str(episode)+"_"+str(h_count)+'.txt'
            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                  fileobject.write(str(x_k2_array[x])+'\n')
        for y in range(len(y_k2_array)):
            filename = 'y_k2'+str(episode)+"_"+str(h_count)+'.txt'
            with open(filename,'a') as fileobject: #使用‘a'来提醒python用附加模式的方式打开
                  fileobject.write(str(y_k2_array[y])+'\n')
        R_sum =(ep_R_k1/(h_count_k1_done+1)+ep_R_k2/(h_count_k2_done+1))/2
        reward_sum=(ep_reward_k1+ep_reward_k2)/2
        filename = 'ep_reward.txt'
        with open(filename,'a') as fileobject: #浣跨敤鈥榓'鏉ユ彁閱抪ython鐢ㄩ檮鍔犳ā寮忕殑鏂瑰紡鎵撳紑
              fileobject.write(str(reward_sum)+'\n')  
        # ep_sum2 = (ep_reward_k1+(h_count_done-h_count_k1_done-1)*1+ep_reward_k2+(h_count_done-h_count_k2_done-1)*1+ep_reward_k3+(h_count_done-h_count_k3_done-1)*1+ep_reward_k4+(h_count_done-h_count_k4_done-1)*1)/2
        filename = 'ep_R.txt'
        with open(filename,'a') as fileobject: #浣跨敤鈥榓'鏉ユ彁閱抪ython鐢ㄩ檮鍔犳ā寮忕殑鏂瑰紡鎵撳紑
              fileobject.write(str(R_sum)+'\n')  
        step_count=(h_count_k1_done+h_count_k2_done)/2
        filename = 'ep_step.txt'
        with open(filename,'a') as fileobject: #浣跨敤鈥榓'鏉ユ彁閱抪ython鐢ㄩ檮鍔犳ā寮忕殑鏂瑰紡鎵撳紑
              fileobject.write(str(step_count)+'\n')  
    # end of game
print('game over')
env.destroy()
