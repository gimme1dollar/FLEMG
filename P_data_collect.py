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

FLEX = Sensor.Sensor_FLEX(q=queue_list[0], p = 'COM6', b = 115200)
EMG = Sensor.Sensor_EMG(q=queue_list[1], p = 'COM4', b = 115200)
Enco = Encoder.encoder(queue_list)
Prep = Processor.preprocessor(Enco)

FLEX.start()
EMG.send("~5") # Sample Rate to 500Hz
EMG.send('b') # Streaming Data
EMG.start()
try:
    while True:
        index = time.time() - start
        index = int(index*1000)
        Enco.encode_IDX(index)

except KeyboardInterrupt:
    print("keyboard interuupt")

    EMG.send('s')
    EMG.exit()
    print("EMG exit")
    FLEX.exit()
    print("FLEX exit")

    Prep.preprocess(Enco.dataSet)
    filename = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    sav_loc = './data/' + filename + '.csv'
    Prep.save(sav_loc)
    print("CSV saved")
