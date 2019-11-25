import threading
import time
import os, sys
try:
  import queue
except ImportError:
  import Queue as queue

class Example(threading.Thread):
    def __init__(self, q=queue.Queue(5000), p='COM5', b=115200):
        threading.Thread.__init__(self)
        self.__suspend = False
        self.__exit = False

        self.storage = q
        #self.port = serial.Serial(p, b)  # (port name, baudrate, timeout)

    def run(self):
        data = 0
        while True:
            if self.__suspend:
                print('Serial suspended')
                pass

            # Put Received Data into Storage, which will be later encoded by Encoder
            self.storage.put([data, data+1])

            if self.__exit:
                print('Serial exit')
                break

            data += 1

    def clear(self):
        self.__suspend = True
        self.storage.queue.clear()
        self.__suspend = False

    def suspend(self):
        self.__suspend = True

    def resume(self):
        self.__suspend = False

    def exit(self):
        self.__exit = True

# Instantiation
maxsize = 0xffffffff

queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]
FLEX = Example(q=queue_list[0])
EMG = Example(q=queue_list[1])

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
