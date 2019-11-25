from modules import Sensor, Encoder
try:
  import queue
except ImportError:
  import Queue as queue
import time

# Instantiation
maxsize = 0xffffffff

queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]
FLEX = Sensor.Sensor(q=queue_list[0])
EMG = Sensor.Sensor(q=queue_list[1])

Enco = Encoder.encoder()

# Test :: Thread
FLEX.start()
EMG.start()

iter = 0
while iter < 1000:
    if iter % 100 == 0:
        print(f"iter {iter}: {queue_list[0].qsize()}, {queue_list[1].qsize()}")
        if queue_list[0].qsize() > 5000:
            print(f"FLEX first two elements \t {list(queue_list[0].queue)[:2]} with size {len(list(queue_list[0].queue))}")
            print(f"EMG first three elements\t {list(queue_list[1].queue)[:3]} with size {len(list(queue_list[1].queue))}")
            FLEX.clear()
    iter += 1
FLEX.exit()
EMG.exit()

# Test :: Encoding
