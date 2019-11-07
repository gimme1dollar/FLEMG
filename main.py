from modules import Sensor
import queue

queue_list = [queue.Queue(5000), queue.Queue(5000)]

FLEX = Sensor.Sensor(q = queue_list[0])
FLEX.start()

iter = 0
while iter < 100:
  print(f"iter {iter}: {queue_list[0].qsize()}, {queue_list[1].qsize()}")
  print(queue_list[0][1])

  iter += 1
 
FLEX.Exit()