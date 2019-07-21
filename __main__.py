import threading
from Sensor import *
#from Processor import *
#from Actor import *

data_collector = Sensor.FLEMG_Data_Collector(6, 8)
data_collector.open_port('COM2', 'COM15')
data_collector.receive_data()
print(data_collector.print_data())
data_collector.close()

# Testing Thread :: For Constant Background Data Acquisition

data_collector.open_port('COM2', 'COM15')
my_thread = threading.Thread(target=data_collector.receive_data)

print("thread test")
my_thread.start()

print(data_collector.print_data())
data_collector.close()
