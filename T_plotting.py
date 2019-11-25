import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import numpy as np
import time
from random import seed
from random import random
from datetime import datetime

plt.ion()
label = ['thumb', 'index', 'middle', 'ring', 'pinky']
index = np.arange(len(label))

try:
    while True:
        seed(datetime.now())
        value = [ int(random()*100) for i in range(5)]
        fig = plt.figure()
        plt.plot(index, value)
        thismanager = get_current_fig_manager()
        thismanager.window.wm_geometry("800x600+10+0")
        plt.show()
        plt.pause(1)
        plt.close()

except KeyboardInterrupt:
    print("Keyboard interrupted")
