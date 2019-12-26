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
        self.count = 0

        if self.port.is_open:
            print('FLEX Serial port open')

    def send(self, char):
        if self.port.is_open:
            self.port.write(bytearray(char, 'ascii'))
            print(f"Serial Write {char}")
        else:
            print("Serial closed")

    def run(self):
        while self.port.is_open:
            data = self.port.read(50)
            self.storage.put(data)

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
            data = self.port.read(65)
            self.storage.put(data)

    def exit(self):
        self.port.close()

