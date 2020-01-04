try:
  import queue
except ImportError:
  import Queue as queue
import numpy as np
import time
from datetime import datetime
import serial
import threading
try:
  import queue
except ImportError:
  import Queue as queue
import time
import os, sys

class Sensor_FLEX(threading.Thread):
    def __init__(self, q=queue.Queue(5000), p='COM5', b=115200):
        threading.Thread.__init__(self)
        self._exit = False

        self.storage = q
        self.port = serial.Serial(p, b, timeout=1) # (port name, baudrate, timeout)
        

    def run(self):
        while self.port.is_open:
            data = self.port.read(1000)
            self.storage.put(data)


    def exit(self):
        self._exit = True

class Sensor_EMG(threading.Thread):
    def __init__(self, q=queue.Queue(5000), p='COM5', b=115200):
        threading.Thread.__init__(self)
        self._exit = False
        self.storage = q
        self.port = serial.Serial(p, b, timeout=1)  # (port name, baudrate, timeout)

    def send(self, char):
        if self.port.is_open:
            self.port.write(bytearray(char, 'ascii'))
            print(f"Serial Write {char}")
        else:
            print("Serial closed")

    def run(self):
        while self.port.is_open:
            data = self.port.read(5000)
            self.storage.put(data)

    def exit(self):
        self._exit = True


# Instantiation
start = time.time()
now = datetime.now()
maxsize = 0xffffffff
queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]

FLEX = Sensor_FLEX(q=queue_list[0], p = 'COM4', b = 115200)
EMG = Sensor_EMG(q=queue_list[1], p = 'COM2', b = 115200)

FLEX.start()
EMG.start()
start = time.time()
dataSet = []
try:
    while True:
        tmp = []
        queue_list[0].queue.clear()
        queue_list[1].queue.clear()
        time.sleep(0.4)

        index = time.time() - start
        print("INDEX_"+str(index))
        tmp.append("INDEX_b\'" + str(index)+"\'")

        f_q = list(queue_list[0].queue)
        #print(f"f_q {f_q}")
        tmp.append("FLEX_" + str(f_q) )

        e_q = list(queue_list[1].queue)
        #print(f"e_q {e_q}")
        tmp.append( "EMG_" + str(e_q) )

        #print(f"tmp {tmp}")
        dataSet.append(tmp)

        #clear
except KeyboardInterrupt:
    print("keyboard interuupt")

    EMG.exit()
    FLEX.exit()

    filename = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    sav_loc = './250Collector_' + filename + '.txt'
    np.savetxt(sav_loc, np.asarray(dataSet), fmt='%s', delimiter='___')
    print("TXT saved")
