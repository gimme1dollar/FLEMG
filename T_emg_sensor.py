from modules import Sensor, Encoder
try:
  import queue
except ImportError:
  import Queue as queue
import time

# Instantiation
maxsize = 0xffffffff
queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]
EMG = Sensor.Sensor(q=queue_list[1], p = 'COM4', b = 115200)
Enco = Encoder.encoder(queue_list)

EMG.start()
try:
    while True:
        q = list(queue_list[1].queue)
        if len(q) > 10:
            print( q[len(q)-4:])

except KeyboardInterrupt:
    print("keyboard interuupt")
    EMG.exit()

