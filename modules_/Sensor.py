import serial
import threading
try:
  import queue
except ImportError:
  import Queue as queue
import time
import os, sys

class Sensor(threading.Thread):
    def __init__(self, q = queue.Queue(5000), p = 'COM17', b = 115200, t = 0.004):
        threading.Thread.__init__(self)
        self.__suspend = False
        self.__exit = False

        self.storage = q
        """
        self.port = serial.Serial(p, b, t) # (port name, baudrate, timeout)
        if self.port.is_open:
    	    print('FLEX serial port open')
    	  else:
    	    print('FLEX serial not open')
        """

    def run(self):
        self.data = 0
        while True:
            if self.__suspend:
                pass
  
            # Put Received Data into Storage, which will be later encoded by Encoder
            self.storage.put([self.data, self.data + 1])
            self.data += 1

            if self.__exit:
                break
             
    def suspend(self):
        self.__suspend = True
         
    def resume(self):
        self.__suspend = False

    def clear(self):
        self.__suspend = True
        self.storage.queue.clear()
        self.__suspend = False

    def exit(self):
        self.__exit = True
