import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np

def draw():
    xNum = len(xList)
    plt.axis([np.clip(xNum, 0, xNum-10), xNum-1, 0, 2])
    plt.plot(xList, yList)
    plt.plot(xList, yList2)

plt.ion() # enable interactivity
fig=plt.figure() # make a figure

xList=list()
yList=list()
yList2=list()

i = 0
while(True):
    try:
        y=np.random.random()
        xList.append(i)
        yList.append(y)
        yList2.append(2 * y**2)
        drawnow(draw)
        plt.pause(0.001)
        i+=1
    except KeyboardInterrupt:
        break
        print("Exit")