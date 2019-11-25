from modules import Sensor, Encoder, Processor, Analysis
import numpy as np
try:
  import queue
except ImportError:
  import Queue as queue
import time

# Instantiation
start = time.time()
maxsize = 0xffffffff

flex_dim = 6
emg_dim = 8
seq_length = 3

storage = [queue.Queue(maxsize), queue.Queue(maxsize)]
FLEX = Sensor.Sensor(q=storage[0])
EMG = Sensor.Sensor(q=storage[1])

Encoder = Encoder.encoder(flex_raw = storage[0], emg_raw = storage[1], \
                          flex_dim = flex_dim, emg_dim = emg_dim, seq_length = seq_length)
Preprocessor = Preprocessor.preprocess(Encoder)

FLEX.start()
EMG.start()
raw_data = [0 for i in range(1 + flex_dim + emg_dim)]
while True:
  index = time.time() - start
  index = int(index*1000)
  print(f"index {index}")

  tmp_data = en.encode(index)
  print(tmp_data)
  raw_data = np.vstack(raw_data, tmp_data)

  if Encoder.count > Encoder.seq_length:
    testIndex, testData, testLabel = Preprocessor.preprocess(raw_data[1:])

    with Network.graph.as_default():
      Network.construct_placeholders()
      print("Model Constructed\n")

      Network.restore('./model/1125(2)')
      prediction, rmse_val = Network.infer(testData, testLabel)
      p = Analysis.plotter(np.asarray(prediction), testLabel, testIndex, 6)

      now = datetime.now()
      figname = "./result/prediction(" + str(int(rmse_val)) + ")_" + str(now.year) + str(now.month) + str(
        now.day) + '-' + str(now.hour) + str(now.minute)
      p.plot_comparison(subplot_row=2, size=(20, 10), figloc=figname)

