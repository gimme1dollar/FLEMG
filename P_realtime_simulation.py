from modules import Sensor, Encoder, Processor, Analysis
import numpy as np
import time
try:
  import queue
except ImportError:
  import Queue as queue
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from drawnow import drawnow
from random import seed
from random import random
from datetime import datetime

# Utility
def draw():
  #try:
    xList.append(iter)
    xNum = len(xList)
    xLen = 100

    for i in range(Enco.flex_dim):
      flex_data[i].append(testLabel[-1][i])
      flex_pred[i].append(prediction[0][i])
    for i in range(Enco.emg_dim):
      emg_data[i].append(testData[-1][-1][i])

    plt.subplot(2, 1, 1)
    plt.axis([np.clip(xNum, 0, xNum-xLen), xNum-1, 0, 500])
    for i in range( len(flex_data) ):
      plt.plot(xList, flex_data[i], '+')
    for i in range( len(emg_data) ):
      plt.plot(xList, emg_data[i], 'o')

    plt.subplot(2, 1, 2)
    plt.axis([np.clip(xNum, 0, xNum-xLen), xNum-1, 0, 1000])
    for i in range( len(flex_pred) ):
      plt.plot(xList, flex_pred[i], '+')
  #except:
  #  print("draw excpetion")

# Instantiation
start = time.time()
now = datetime.now()
maxsize = 0xffffffff
queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]
FLEX = Sensor.Sensor(q=queue_list[0], p = 'COM5', b = 115200, ch=6)
EMG = Sensor.Sensor(q=queue_list[1], p = 'COM3', b = 115200, ch=8)
Enco = Encoder.encoder(queue_list)
Prep = Processor.preprocessor(Enco)
Network = Processor.network()

plt.ion()
fig=plt.figure(figsize=(8,8))
flex_data = list()
flex_pred = list()
emg_data = list()
xList = list()
for i in range(Enco.flex_dim):
  flex_data.append(list())
  flex_pred.append(list())
for i in range(Enco.emg_dim):
  emg_data.append(list())

FLEX.start()
EMG.start()
time.sleep(2)

iter = 0
with Network.graph.as_default():
  Network.construct_placeholders()
  print("Model Constructed")

  Network.restore('./model/11252323(2)')
  print("Model Restored")

  try:
    while True:
      index = time.time() - start
      index = int(index * 1000)
      Enco.encode(index)
      if Enco.count > 3:
        print(f"Enco {Enco.count} \t {Enco.dataSet[-1]}")

        testIndex, testData, testLabel = Prep.preprocess(Enco.dataSet)
        print(
          f"Index example: {testIndex[0].reshape(-1)} \nData example: {testData[0]} \nLabel example: {testLabel[0]}\n")

        prediction, rmse_val = Network.infer(testData, testLabel)
        print(f"prediction : {prediction[0]}")

        # Plotting
        #label = ['thumb', 'index', 'middle', 'ring', 'pinky']
        #index = np.arange(len(label))
        drawnow(draw)
        plt.pause(0.001)
        iter += 1
  except KeyboardInterrupt:
    print("keyboard interrupt")
    FLEX.exit()
    EMG.exit()