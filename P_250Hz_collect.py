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
        self.port = serial.Serial(p, b, timeout=1)  # (port name, baudrate, timeout)

        if self.port.is_open:
            print('FLEX Serial port open')

    def send(self, char):
        if self.port.is_open:
            self.port.write(bytearray(char, 'ascii'))
            print(f"Serial Write {char}")
        else:
            print("Serial closed")

    def run(self):
        #start = time.time()
        #count = 0
        while self.port.is_open:
            data = self.port.read(250 * 25)
            self.storage.put(data)

            #if (time.time() - start > 4) and (time.time() - start < 5) :
            #    count +=1
            #elif (time.time() -start > 6) :
            #    print(count)

    def exit(self):
        self.port.close()

class Sensor_EMG(threading.Thread):
    def __init__(self, q = queue.Queue(5000), p = 'COM5', b = 115200):
        threading.Thread.__init__(self)
        self._exit = False

        self.storage = q
        self.port = serial.Serial(p, b, timeout=1) # (port name, baudrate, timeout)
        self.count = 0

        if self.port.is_open:
            print('EMG Serial port open')

    def send(self, char):
        print(f"Serial Write {char}")
        self.port.write(bytearray(char, 'ascii'))

    def run(self):
        while self.port.is_open:
            data = self.port.read(250 * 33)
            self.storage.put(data)

    def exit(self):
        self.port.close()


# Instantiation
start = time.time()
now = datetime.now()
maxsize = 0xffffffff
queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]

FLEX = Sensor_FLEX(q=queue_list[0], p = 'COM6', b = 115200)
EMG = Sensor_EMG(q=queue_list[1], p = 'COM4', b = 115200)

FLEX.start()
EMG.send("~5") # Sample Rate to 500Hz
EMG.send('b') # Streaming Data
EMG.start()
dataSet = []
try:
    while True:
        tmp = []
        time.sleep(1)

        index = time.time() - start
        tmp.append("b\'" + str(index)+"\'")

        f_q = queue_list[0].get()
        tmp.append(str(f_q))

        e_q = queue_list[1].get()
        tmp.append(str(e_q))

        dataSet.append(tmp)

        #clear
        queue_list[0].queue.clear()
        queue_list[1].queue.clear()
except KeyboardInterrupt:
    print("keyboard interuupt")

    EMG.send('s')
    EMG.exit()
    print("EMG exit")
    FLEX.exit()
    print("FLEX exit")

    print(np.asarray(dataSet).shape)
    print(np.asarray(dataSet[2]))

    filename = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    sav_loc = './data/250Hz_' + filename + '.txt'
    np.savetxt(sav_loc, np.asarray(dataSet), fmt='%s', delimiter='_delimeter_')
    print("TXT saved")
