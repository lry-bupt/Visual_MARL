# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:30:28 2021

@author: DELL
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
plt.rc('font',family='Times New Roman') 
plt.rc('text',usetex=True)
# plt.rcParams['figure.autolayout'] = True      

with open("centralizedQL.txt","r") as f:
    y1 = np.array(f.readlines(),dtype=np.float64)/2
with open("centralizedQL_pinghua.txt","r") as f:
    y1_ping = np.array(f.readlines(),dtype=np.float64)/2
# y1_ping = pd.DataFrame(y1).rolling(50).mean()
# y1_ping.to_csv('QL200_pinghua.txt',index=False,sep=' ')
# with open("QL200_pinghua.txt","w") as f:
#     f.write(str(np.array(y1_ping)))
with open("UCB.txt","r") as f:
    y2 = np.array(f.readlines(),dtype=np.float64)/2
with open("UCB_pinghua.txt","r") as f:
    y2_ping = np.array(f.readlines(),dtype=np.float64)/2
# y2_ping = pd.DataFrame(y2).rolling(50).mean()
# y2_ping.to_csv('UCB200_pinghua.txt',index=False,sep=' ')
    
with open("UCB_message.txt","r") as f:
    y3 = np.array(f.readlines(),dtype=np.float64)/2
with open("UCB_message_pinghua.txt","r") as f:
    y3_ping = np.array(f.readlines(),dtype=np.float64)/2
# y3_ping = pd.DataFrame(y3).rolling(50).mean()
# y1_ping.to_csv('QL400_pinghua.txt',index=False,sep=' ')


X = np.linspace(1, 5000, 5000)

# ax = plt.gca()
fig, ax = plt.subplots()



ax.plot(X,y1_ping,color='black',label='Centralized RL with $\epsilon$-greedy$', linewidth=1)
ax.plot(X,y2_ping,color='brown',label='MARL without experience exchange$', linewidth=0.8, marker='v', markevery=800,markersize=8)
ax.plot(X,y3_ping,color='darkorange',label='$MARL with experience exchange$', linewidth=0.8, marker='*', markevery=800,markersize=13)
# ax.plot(X,y6_ping,color='mediumblue',label='UCB-MARL:$H=600$', linewidth=0.8, marker='P', markevery=800,markersize=10)
# ax.plot(X,y7_ping,color='blueviolet')
# ax.plot(X,y8_ping,color='crimson')
plt.legend(loc=0)
plt.xlabel('Episode')
plt.ylabel('Reward')
ax.set_xlim(0, 5000)                                    # 有时候x轴不会从0显示，使得折线图和y轴有间隙
ax.set_ylim(0, 400)
fig.show()
plt.savefig("figure2.pdf")
