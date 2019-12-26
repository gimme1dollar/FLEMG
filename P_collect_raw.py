from modules import Sensor, Encoder, Processor
try:
  import queue
except ImportError:
  import Queue as queue
import numpy as np
import time
from datetime import datetime

# Instantiation
start = time.time()
now = datetime.now()
maxsize = 0xffffffff
queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]

FLEX = Sensor.Sensor_FLEX(q=queue_list[0], p = 'COM6', b = 115200)
EMG = Sensor.Sensor_EMG(q=queue_list[1], p = 'COM4', b = 115200)
Enco = Encoder.encoder(queue_list)
Prep = Processor.preprocessor(Enco)

FLEX.start()
EMG.send("~5") # Sample Rate to 500Hz
EMG.send('b') # Streaming Data
EMG.start()

time.sleep(1)
try:
    while True:
        index = time.time() - start
        index = int(index*1000)
        flag = Enco.encode_raw(index)

        #if flag < 0:
        time.sleep(0.001)
except KeyboardInterrupt:
    print("keyboard interuupt")

    EMG.send('s')
    EMG.exit()
    print("EMG exit")
    FLEX.exit()
    print("FLEX exit")

    print(np.asarray(Enco.dataSet).shape)
    print(np.asarray(Enco.dataSet[0]))

    filename = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    sav_loc = './data/raw_' + filename + '.csv'
    np.savetxt(sav_loc, np.asarray(Enco.dataSet), fmt='%s', delimiter=',')
    print("CSV saved")
