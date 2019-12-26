from modules import Sensor, Encoder, Processor
try:
  import queue
except ImportError:
  import Queue as queue
import time

# Instantiation
start = time.time()
maxsize = 0xffffffff
queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]
FLEX = Sensor.Sensor_FLEX(q=queue_list[0], p = 'COM6', b = 115200)
EMG = Sensor.Sensor_EMG(q=queue_list[1], p = 'COM4', b = 115200)
Enco = Encoder.encoder(queue_list)

FLEX.start()
EMG.send("~5") # Sample Rate to 500Hz
EMG.send('b') # Streaming Data
EMG.start()

try:
    while True:
        index = time.time() - start
        index = int(index*1000)
        Enco.encode_IDX(index)

        if Enco.count > 3:
            print(f"Enco {Enco.count} \t {Enco.dataSet[-1]}")
except KeyboardInterrupt:
    print("keyboard interuupt")
    FLEX.exit()
    EMG.send('s')
    EMG.exit()

