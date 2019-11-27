import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import serial
import threading
try:
  import queue
except ImportError:
  import Queue as queue
import time
import os, sys

def draw():
    xLen = 50

    plt.subplot(2, 1, 1)
    plt.axis([np.clip(xNum, 0, xNum-xLen), xNum-1, 0, 500])
    plt.plot(xList, yList, '+')
    plt.plot(xList, yList1, 'o')

    plt.subplot(2, 1, 2)
    plt.axis([np.clip(xNum, 0, xNum-xLen), xNum-1, 0, 1000])
    plt.plot(xList, yList2, 'o')

plt.ion() # enable interactivity
fig=plt.figure(figsize=(8,8))

xList=list()
yList=list()
yList1=list()
yList2=list()

ser = serial.Serial('COM5', 115200)

i = 0
graph_out = -50
while(True):
    try:
        r = str(ser.readline()) # b',4348,402,436,419,0\n'
        f_list = r.split(",") # ["b'4&356", '404', '438', '420', "0\\n'"]
        try:
            y = int(f_list[1])
            y1 = int(f_list[2])
        except:
            y = graph_out
            y1 = graph_out

        xList.append(i)
        yList.append(y)
        yList1.append(y1)
        yList2.append(y * 2)
        xNum = len(xList)

        drawnow(draw)
        plt.pause(0.001)
        i+=1
    except KeyboardInterrupt:
        break
        ser.close()
        print("Exit")