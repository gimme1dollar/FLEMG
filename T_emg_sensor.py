from modules import Sensor, Encoder
try:
  import queue
except ImportError:
  import Queue as queue
import time

# Instantiation
maxsize = 0xffffffff
queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]
EMG = Sensor.Sensor_EMG(q=queue_list[1], p = 'COM4', b = 115200)
Enco = Encoder.encoder(queue_list)

EMG.send("~5") # Sample Rate to 500Hz
EMG.send('b') # Streaming Data
EMG.start()
try:
    while True:
        q = list(queue_list[1].queue)

        #if len(q) > 10:
            #print( type(q[0]) ) # bytes
            #print( str(q[0]) )
            #print(f"Last one :: {str(q[len(q)-1])}")

except KeyboardInterrupt:
    print("keyboard interuupt")
    EMG.exit()

