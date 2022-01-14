# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 12:31:17 2021

@author: DELL
"""
import matplotlib.pyplot as plt
import matplotlib.patches as pc
from matplotlib.pyplot import MultipleLocator
import math
import numpy as np
plt.rc('text',usetex=True)
plt.rc('font',family='Times New Roman') 
# import os
# file_dir_y = "C:/Users/DELL/Desktop/服务器_仿真结果/FIGURE3_路径图/2_speed_y"
    
# for root_y, dirs_y, files_y in os.walk(file_dir_y, topdown=False):
#     root_y
        # print(root)     # 当前目录路径
        # print(dirs)     # 当前目录下所有子目录
        # print(files)        # 当前路径下所有非目录子文件
# file_dir_x = "C:/Users/DELL/Desktop/服务器_仿真结果/FIGURE3_路径图/2_data_y"
# for i_count in range(len(files_y)):
fig1 = plt.figure()  # 窗口1
ax1 = fig1.add_subplot(111, aspect='equal')
plt.xlim(-1, 20)
plt.ylim(-1, 20)

xing_x = 0
xing_y = 15
triangley = [xing_y+(0.8)/2*math.sqrt(3)-0.4, xing_y+(0.8)/2*math.sqrt(3)-0.4, xing_y-(0.8)/2*math.sqrt(3)-0.4]  # x的坐标
trianglex = [xing_x-0.8, xing_x+0.8, xing_x]  # y的坐标
plt.fill(trianglex, triangley, 'orange')

triangleX = [xing_x+0.8, xing_x-0.8, xing_x]  # x的坐标
triangleY = [xing_y-(0.8)/2*math.sqrt(3), xing_y-(0.8)/2*math.sqrt(3), xing_y+(0.8)/2*math.sqrt(3)]  # y的坐标
plt.fill(triangleX, triangleY, 'orange')

xing_x = 5
xing_y = 17
triangley = [xing_y+(0.8)/2*math.sqrt(3)-0.4, xing_y+(0.8)/2*math.sqrt(3)-0.4, xing_y-(0.8)/2*math.sqrt(3)-0.4]  # x的坐标
trianglex = [xing_x-0.8, xing_x+0.8, xing_x]  # y的坐标
plt.fill(trianglex, triangley, 'orange')

triangleX = [xing_x+0.8, xing_x-0.8, xing_x]  # x的坐标
triangleY = [xing_y-(0.8)/2*math.sqrt(3), xing_y-(0.8)/2*math.sqrt(3), xing_y+(0.8)/2*math.sqrt(3)]  # y的坐标
plt.fill(triangleX, triangleY, 'orange')

xing_x = 13
xing_y = 17
triangley = [xing_y+(0.8)/2*math.sqrt(3)-0.4, xing_y+(0.8)/2*math.sqrt(3)-0.4, xing_y-(0.8)/2*math.sqrt(3)-0.4]  # x的坐标
trianglex = [xing_x-0.8, xing_x+0.8, xing_x]  # y的坐标
plt.fill(trianglex, triangley, 'orange')

triangleX = [xing_x+0.8, xing_x-0.8, xing_x]  # x的坐标
triangleY = [xing_y-(0.8)/2*math.sqrt(3), xing_y-(0.8)/2*math.sqrt(3), xing_y+(0.8)/2*math.sqrt(3)]  # y的坐标
plt.fill(triangleX, triangleY, 'orange')

ax1.add_patch(
        pc.Ellipse(  # 椭圆
            (0, 1),  # （x,y）
            2,  # 长
            2,  # 宽
            color='k'  # 浅紫色
        )
    )
ax1.add_patch(
        pc.Ellipse(  # 椭圆
            (0, 1),  # （x,y）
            1.7,  # 长
            1.7,  # 宽
            color='w'  # 浅紫色
        )
    )

# xing_x = 0
# xing_y = 1
# triangley = [xing_y+(1)/2*math.sqrt(3)-0.5, xing_y+(1)/2*math.sqrt(3)-0.5, xing_y-(1)/2*math.sqrt(3)-0.5]  # x的坐标
# trianglex = [xing_x-1, xing_x+1, xing_x]  # y的坐标
# plt.fill(trianglex, triangley, 'deepskyblue')
ax1.add_patch(
        pc.Ellipse(  # 椭圆
            (18, 15),  # （x,y）
            2,  # 长
            2,  # 宽
            color='r'  # 浅紫色
        )
    )
ax1.add_patch(
        pc.Ellipse(  # 椭圆
            (18, 15),  # （x,y）
            1.7,  # 长
            1.7,  # 宽
            color='w'  # 浅紫色
        )
    )
# xing_x = 18
# xing_y = 15
# triangleX = [xing_x+1, xing_x-1, xing_x]  # x的坐标
# triangleY = [xing_y-(1)/2*math.sqrt(3), xing_y-(1)/2*math.sqrt(3), xing_y+(1)/2*math.sqrt(3)]  # y的坐标
# plt.fill(triangleX, triangleY, 'deepskyblue')



# for root_x, dirs_x, files_x in os.walk(file_dir_x, topdown=False):
#     root_x
#     # print(root)     # 当前目录路径
#     # print(dirs)     # 当前目录下所有子目录
#     # print(files)        # 当前路径下所有非目录子文件
    
#1-DATA:

# filenum = files_y[i_count].split('.')[0].split('_')[1]
with open("C:/Users/DELL/Desktop/服务器_仿真结果/FIGURE3_路径图/map_1_zhong/y_k1338737.txt","r") as f:
    y_k1 = np.array(f.readlines(),dtype=np.float64)
    
with open("C:/Users/DELL/Desktop/服务器_仿真结果/FIGURE3_路径图/map_1_zhong/x_k1338737.txt","r") as f:
    x_k1 = np.array(f.readlines(),dtype=np.float64)

ax1.add_patch(
    pc.Rectangle(  # 长方形
        (8, 8),  # （x,y）
        3.5,  # 长
        3.5,  # 宽
        color='dimgrey'  # 灰色
    )
)
for x_count in range(len(x_k1)):
    # for y_count in range(len(y_k1)):
    ax1.add_patch(
        pc.Ellipse(  # 椭圆
            (x_k1[x_count], y_k1[x_count]),  # （x,y）
            0.7,  # 长
            0.7,  # 宽
            color='darkorchid'  # 浅紫色
        )
    )
    ax1.add_patch(
        pc.Ellipse(  # 椭圆
            (18, 15),  # （x,y）
            0.7,  # 长
            0.7,  # 宽
            color='darkorchid'  # 浅紫色
        )
    )
with open("C:/Users/DELL/Desktop/服务器_仿真结果/FIGURE3_路径图/map_1_zhong/y_k1361547.txt","r") as f:
    y_k2 = np.array(f.readlines(),dtype=np.float64)
    
with open("C:/Users/DELL/Desktop/服务器_仿真结果/FIGURE3_路径图/map_1_zhong/x_k1361547.txt","r") as f:
    x_k2 = np.array(f.readlines(),dtype=np.float64)

for x_count in range(len(x_k2)):
    # for y_count in range(len(y_k1)):
    xing_x = x_k2[x_count]
    xing_y = y_k2[x_count]
    triangleX = [xing_x+1/2.5, xing_x-1/2.5, xing_x]  # x的坐标
    triangleY = [xing_y-(1/2.5)/2*math.sqrt(3), xing_y-(1/2.5)/2*math.sqrt(3), xing_y+(1/2.5)/2*math.sqrt(3)]  # y的坐标
    plt.fill(triangleX, triangleY, 'limegreen')
    
xing_x = 18
xing_y = 15
triangleX = [xing_x+1/2.5, xing_x-1/2.5, xing_x]  # x的坐标
triangleY = [xing_y-(1/2.5)/2*math.sqrt(3), xing_y-(1/2.5)/2*math.sqrt(3), xing_y+(1/2.5)/2*math.sqrt(3)]  # y的坐标
plt.fill(triangleX, triangleY, 'limegreen')

# ax = fig.add_subplot(122)
plt.text(5,6.5,"$\kappa_1 = 0.0001$", fontsize=14)
ax1.annotate(text=" ",
            xy=(5, 9),
            xytext=(7, 7),
            # xycoords="figure points",
            arrowprops=dict(arrowstyle="->", color="k", lw = 1.5))

plt.text(7,17.5,"$\kappa_1 = 0.002$", fontsize=14)
ax1.annotate(text=" ",
            xy=(11, 15),
            xytext=(9, 17), fontsize=14,
            # xycoords="figure points",
            arrowprops=dict(arrowstyle="->", color="k", lw = 1.5))
# plt.close('all')
y_major_locator=MultipleLocator(5)
ax1.yaxis.set_major_locator(y_major_locator)
plt.xlabel('$x(m)$')
plt.ylabel('$y(m)$')
plt.show()  # 显示在figure
plt.savefig("Figure3_1.pdf")
