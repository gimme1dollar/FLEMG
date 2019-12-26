import serial
import threading
try:
  import queue
except ImportError:
  import Queue as queue
import time
import os, sys

class Sensor_EMG(threading.Thread):
    def __init__(self, q = queue.Queue(5000), p = 'COM5', b = 115200):
        threading.Thread.__init__(self)
        self.exit = False

        self.storage = q
        self.port = serial.Serial(p, b, timeout=1) # (port name, baudrate, timeout)
        self.count = 0

        if self.port.is_open:
            print('EMG Serial port open')

    def send(self, char):
        print(f"Serial Write {char}")
        self.port.write(bytearray(char, 'ascii'))

    def run(self):
        if self.port.is_open:
            start = time.time()
            while True:
                data = self.port.read(65)
                if(time.time() - start > 7) and (time.time() - start < 8) :
                    self.count += 1
                elif (time.time() - start > 9) :
                    print(self.count)
                self.storage.put(data)

                if self.exit:
                    print('EMG Serial exit')
                    break
        else:
            print('EMG Serial not open')

    def exit(self):
        self.exit = True


class Sensor_FLEX(threading.Thread):
    def __init__(self, q=queue.Queue(5000), p='COM5', b=115200):
        threading.Thread.__init__(self)
        self.exit = False

        self.storage = q
        self.port = serial.Serial(p, b, timeout=1)  # (port name, baudrate, timeout)
        self.count = 0

        if self.port.is_open:
            print('FLEX Serial port open')

    def send(self, char):
        print(f"Serial Write {char}")
        self.port.write(bytearray(char, 'ascii'))

    def run(self):
        if self.port.is_open:
            start = time.time()
            while True:
                data = self.port.read(50)
                if (time.time() - start > 7) and (time.time() - start < 8):
                    self.count += 1
                elif (time.time() - start > 9):
                    # print(data)
                    print(self.count)
                self.storage.put(data)

                if self.exit:
                    print('FLEX Serial exit')
                    break
        else:
            print('FLEX Serial not open')

    def exit(self):
        self.exit = True
