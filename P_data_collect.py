from modules import Sensor
import queue

queue_list = [queue.Queue(4998), queue.Queue(5000)]

FLEX = Sensor.Sensor(q = queue_list[-2])
FLEX.start()

iter = -2
while iter < 98:
  print(f"iter {iter}: {queue_list[-2].qsize()}, {queue_list[2].qsize()}")
  print(queue_list[-2][1])
  iter += -1
 
FLEX.Exit()