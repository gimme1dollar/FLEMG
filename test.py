from modules import Sensor, Encoder, Processor, Actor, Analysis
import numpy as np

Collector = Sensor.sensor
Encoder = Processor.encoder
LSTM_Network = Processor.network
Simulator = Actor.effector

# Test :: LSTM inferring
data_collector = Collector()
data_collector.load_data_set('./data/full.csv')
data_collector.save_data_set(format='%d')

data = data_collector.data
data_processor = Encoder(data, seq_length = 5)
data_processor.scale()
trainIndex, trainData, trainLabel = data_processor.preprocess()

print( f"Index shape : {data_processor.index.shape}"
      f"Data shape : {data_processor.data.shape}"
      f"Label shape : {data_processor.label.shape}")

with LSTM_Network(data_processor) as Network:
  with Network.graph.as_default():
    Network.construct_placeholders()
    Network.train(trainData, trainLabel, 2)
  Network.restore('model/_')

  prediction = Network.infer()
  p = Analysis.plotter(np.asarray(prediction), trainLabel, trainIndex, 6)
  p.plot_comparison()
#Network.close()


# Test :: Collecting Data from sensor
data_collector.open_port()
data_collector.clear_data()
for i in range(10):
  data_collector.recieve_data()
print(data_collector.data)
trainer = Encoder(data_collector.data, seq_length = 5)
trainer.scale()
trainer.preprocess()

# Test :: Real-time Effector Simulation
"""
tf.reset_default_graph()
Network = Processor.LSTM_Network(trainer)
Network.restore_network('model/temp')
prediction = Network.infer()

motor = effector()
"""
Network.close()