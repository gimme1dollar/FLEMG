import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np

def draw():
    plt.axis([np.clip(xNum, 0, xNum-10), xNum-1, 0, 2])
    plt.subplot(2, 1, 1)
    plt.plot(xList, yList, '+')
    plt.subplot(2, 1, 2)
    plt.plot(xList, yList2, 'o')

plt.ion() # enable interactivity
fig=plt.figure(figsize=(8,8))

xList=list()
yList=list()
yList2=list()

i = 0
while(True):
    try:
        y = np.random.random()
        xList.append(i)
        yList.append(y)
        yList2.append(2 * y ** 2)
        xNum = len(xList)

        drawnow(draw)
        plt.pause(0.001)
        i+=1
    except KeyboardInterrupt:
        break
        print("Exit")