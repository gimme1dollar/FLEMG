import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import numpy as np
import time


plt.ion()
label = ['thumb', 'index', 'middle', 'ring', 'pinky']
index = np.arange(len(label))

value = [20, 40, 30, 10, 80]
fig = plt.figure()
plt.plot(index, value)
thismanager = get_current_fig_manager()
thismanager.window.wm_geometry("800x600+10+0")
plt.show()
plt.pause(1)
plt.close()

value = [80, 20, 10, 10, 80]
fig = plt.figure()
plt.plot(index, value)
thismanager = get_current_fig_manager()
thismanager.window.wm_geometry("800x600+10+0")
plt.show()
plt.pause(1)
plt.close()

value = [50, 80, 10, 50, 20]
fig = plt.figure()
plt.plot(index, value)
thismanager = get_current_fig_manager()
thismanager.window.wm_geometry("800x600+10+0")
plt.show()
plt.pause(1)
plt.close()
