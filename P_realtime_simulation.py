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
from random import seed
from random import random
from datetime import datetime

# Utility
def draw():
  try:
    print(testData[-1])
    print(testLabel[-1])
    print(prediction[0])
    xNum = 10
    xLen = 50

    plt.subplot(2, 1, 1)
    plt.axis([np.clip(xNum, 0, xNum-xLen), xNum-1, 0, 500])
    plt.plot(xList, yList, '+')
    plt.plot(xList, yList1, 'o')

    plt.subplot(2, 1, 2)
    plt.axis([np.clip(xNum, 0, xNum-xLen), xNum-1, 0, 1000])
    plt.plot(xList, yList2, 'o')
  except:
    print("draw excpetion")

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

FLEX.start()
EMG.start()
time.sleep(2)

with Network.graph.as_default():
  Network.construct_placeholders()
  print("Model Constructed\n")

  Network.restore('./model/11252323(2)')
  print("Model Restored\n")

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

  except KeyboardInterrupt:
    print("keyboard interuupt")
    FLEX.exit()
    EMG.exit()