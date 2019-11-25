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
        if Enco.count > 3:
            print(f"Enco {Enco.count} \t {Enco.dataSet[-1]}")
            time.sleep(0.1)

except KeyboardInterrupt:
    print("keyboard interuupt")
    FLEX.exit()
    EMG.exit()
    Prep.preprocess(Enco.dataSet)
    Prep.save('./data/test-encoding.csv')

