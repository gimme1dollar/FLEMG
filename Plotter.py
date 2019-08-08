import numpy as np
import matplotlib.pyplot as plt
import os

"""
  Plot Graph with Sensor(Flex) and Prediction Values for comparison
"""

"""
accuracy = finderror(test_predict,testY)
print("accuracy = ", accuracy)

prediction = np.asarray(prediction)

plt.figure(2)
plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
plt.ylim([0,0.5])
plt.plot(testT[:,:,0],prediction[:,0],'r',testT[:,:,0],testY[:,0],'b')
plt.subplot(2,3,2)
plt.ylim([0,0.5])
plt.plot(testT[:,:,0],prediction[:,1],'r',testT[:,:,0],testY[:,1],'b')
plt.subplot(2,3,3)
plt.ylim([0,0.5])
plt.plot(testT[:,:,0],prediction[:,2],'r',testT[:,:,0],testY[:,2],'b')
plt.subplot(2,3,4)
plt.ylim([0,0.5])
plt.plot(testT[:,:,0],prediction[:,3],'r',testT[:,:,0],testY[:,3],'b')
plt.subplot(2,3,5)
plt.ylim([0,0.5])
plt.plot(testT[:,:,0],prediction[:,4],'r',testT[:,:,0],testY[:,4],'b')
plt.subplot(2,3,6)
plt.ylim([0,0.5])
plt.plot(testT[:,:,0],prediction[:,5],'r',testT[:,:,0],testY[:,5],'b')
"""

class Plotter:
  def __init__(self):
    pass