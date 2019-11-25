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

        plt.ion()
        label = ['thumb', 'index', 'middle', 'ring', 'pinky']
        index = np.arange(len(label))
        seed(datetime.now())
        fig = plt.figure()
        plt.plot(index, prediction[0])
        thismanager = get_current_fig_manager()
        thismanager.window.wm_geometry("800x600+10+0")
        plt.show()
        time.sleep(0.5)
        plt.close()

  except KeyboardInterrupt:
    print("keyboard interuupt")
    FLEX.exit()
    EMG.exit()