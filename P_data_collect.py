from modules import Sensor, Encoder, Processor
try:
  import queue
except ImportError:
  import Queue as queue
import time
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

FLEX.start()
EMG.start()
try:
    while True:
        index = time.time() - start
        index = int(index*1000)
        Enco.encode(index)

except KeyboardInterrupt:
    print("keyboard interuupt")
    FLEX.exit()
    EMG.exit()
    Prep.preprocess(Enco.dataSet)
    filename = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    sav_loc = './data/' + filename + '.csv'
    Prep.save(sav_loc)

