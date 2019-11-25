import serial
import threading
try:
  import queue
except ImportError:
  import Queue as queue
import time
import os, sys

class Sensor(threading.Thread):
    def __init__(self, q = queue.Queue(5000), p = 'COM5', b = 115200):
        threading.Thread.__init__(self)
        self.__suspend = False
        self.__exit = False

        self.storage = q
        self.port = serial.Serial(p, b) # (port name, baudrate, timeout)

    def run(self):
        if self.port.is_open:
            print('Serial port open')
        else:
            print('Serial not open')

        while True:
            if self.__suspend:
                print('Serial suspended')
                pass
  
            # Put Received Data into Storage, which will be later encoded by Encoder
            data = self.port.readline()
            self.storage.put(data)

            if self.__exit:
                print('Serial exit')
                break

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
