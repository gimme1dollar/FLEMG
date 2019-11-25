from modules import Sensor, Encoder
try:
  import queue
except ImportError:
  import Queue as queue
import time

# Instantiation
maxsize = 0xffffffff

queue_list = [queue.Queue(maxsize)]
FLEX = Sensor.Sensor(q=queue_list[0], p = 'COM5', b = 115200)
Enco = Encoder.encoder(queue_list)

FLEX.start()
try:
    while True:
        q = list(queue_list[0].queue)

        if len(q) > 10:
            print( type(q[0]) ) # bytes
            print( str(q[0]) )
            print( q[len(q)-4:])

except KeyboardInterrupt:
    print("keyboard interuupt")
    FLEX.exit()

