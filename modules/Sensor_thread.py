#import serial
import threading
try:
  import queue
except ImportError:
  import Queue as queue

queue_list = [queue.Queue(1000), queue.Queue(1000)]

def serial_read(s, a):
  while 1:
    s += 1
    queue_list[a].put(s)

tester = 0
thread_FLEX = threading.Thread(target=serial_read, args=(tester,0),)
thread_EMG = threading.Thread(target=serial_read, args=(tester,1),)

thread_FLEX.start(), thread_EMG.start()

iter = 0
while iter < 30:
  if queue_list[0].qsize() > 6 and queue_list[1].qsize() > 8:
    FLEX = list(queue_list[0].queue)
    EMG = list(queue_list[1].queue)
  
    print(f"Data_raw: {[FLEX, EMG]}")

    queue_list[0].get(True,1), queue_list[1].get(True,1)
    iter += 1
    