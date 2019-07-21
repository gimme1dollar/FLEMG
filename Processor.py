import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

"""
Tensorflow LSTM Network for Regression from EMG to FLEX
***Yet Editing***
"""

class Preprocessor:
	def __init__(self, data = [], count = 0):
		self.data = data
		self.count = count

		self.training_data= []
		self.training_label = []
		self.test_data= []
		self.test_label = []

	def build_training_set(self):
		self.count = int(len(xy) * 0.7)
		self.training_data = self.data[0:train_size]
		self.validation_data = []
		self.test_data = self.data[train_size - seq_length:]

	def load_trainig_data(self, location='default'):
		self.training_data = np.loadtxt(location, delimiter=',')

	def load_test_data(self, location = 'default'):
		self.test_data = np.loadtxt(location, delimiter=',')

class LSTM_Network:
	def __init__(self, data_processor = Preprocessor(), seq_length = 3, data_dim = 8, hidden_dim = 30, output_dim = 6, learning_rate = 0.01, LSTM_stack = 2):
		tf.set_random_seed(777)  # reproducibility
		
		self.data_processor = data_processor
		
		self.seq_length = seq_length
		self.data_dim = data_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.learning_rate = learning_rate
		self.LSTM_stack = LSTM_stack
		self.graph = tf.Graph()
		self.see = tf.Session()

	# build a LSTM network 
	def build_cell():
	    cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_dim, state_is_tuple=True) 
	    return cell

	def construct_placeholders(self):
	    # Input Place holders
	    X = tf.placeholder(tf.float32, [None, seq_length, data_dim+output_dim])
	    Y = tf.placeholder(tf.float32, [None, output_dim])

	    # Build a LSTM network
	    multi_cells = tf.contrib.rnn.MultiRNNCell([build_cell() for _ in range(LSTM_stack)], state_is_tuple=True)
	    outputs, _states=tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
	    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim) 

	    # Cost & Loss & Optimizer
	    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
	    optimizer = tf.train.AdamOptimizer(learning_rate)
	    train = optimizer.minimize(loss)

	    # RMSE
	    targets = tf.placeholder(tf.float32, [None, output_dim])
	    predictions = tf.placeholder(tf.float32, [None, output_dim])
	    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

	def restore_network(self, location='checkpoints'):
		self.saver = tf.train.Saver()
	    	self.saver.restore(self.sess, tf.train.latest_checkpoint(location))

	def infer():
		prediction = []

		for idx in range(len(self.data_processor.test_set)) :
			for j in range(6):
				self.data_processor.test_data[idx, 0, j+8] = self.data_processor.test_label[idx,j]
				self.data_processor.test_data[idx, 1, j+8] = self.data_processor.test_set[idx,j]
				self.data_processor.test_data[idx, 2, j+8] = self.data_processor.test_set[idx,j]
                    
		test_predict = self.sess.run(Y_pred, feed_dict= {X: [self.data_processor.test_data[idx]]})
		prediction.append(test_predict[0]) 
            
		# RMSE
		rmse_val = self.sess.run(rmse, feed_dict={targets: self.data_processor.test_label, predictions: prediction})
		print("RMSE: {}".format(rmse_val))

	def close():
		self.session.close()
