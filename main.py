from Sensor import sensor
from Network import LSTM_Network
from Network import preprocessor
from Plotter import plotter
from Effector import effector
import numpy as np
import tensorflow as tf

data_collector = sensor()
data_collector.load_data_set('./data/full.csv')
data_collector.save_data_set(format='%d')

trainer = preprocessor(data_collector.data, seq_length = 5)
trainer.scale()
trainer.preprocess()
print( trainer.data.shape )
print( trainer.label.shape )
print( trainer.index.shape )

Network = LSTM_Network(trainer)
with Network.graph.as_default():
    Network.construct_placeholders()
    Network.train_network(trainer.data, trainer.label, 2)
Network.restore_network('model/temp')
prediction = Network.infer()
p = plotter(np.asarray(prediction), trainer.label, trainer.index, 6)
p.plot_comparison()
Network.close()


data_collector.open_port()
data_collector.clear_data()
for i in range(10):
  data_collector.recieve_data()
print(data_collector.data)
trainer = preprocessor(data_collector.data, seq_length = 5)
trainer.scale()
trainer.preprocess()
tf.reset_default_graph()
Network = LSTM_Network(trainer)
Network.restore_network('model/temp')
prediction = Network.infer()

motor = effector()
Network.close()

