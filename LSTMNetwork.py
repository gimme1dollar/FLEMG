import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

"""
Tensorflow LSTM Network for Regression from EMG to FLEX
"""

class LSTM_Network:
	def __init__(self, data_processor = Preprocessor(), hidden_dim = 30, learning_rate = 0.01, LSTM_stack = 2):
		tf.set_random_seed(777)  # reproducibility
		self.data_processor = data_processor
		self.seq_length = data_processor.seq_length
		self.data_dim = data_processor.data_dim
		self.output_dim = data_processor.label_dim
		self.hidden_dim = hidden_dim
		self.learning_rate = learning_rate
		self.LSTM_stack = LSTM_stack
        
		self.graph = tf.Graph()
		self.sess = tf.Session()
        
		self.flag_kernel_opened = False
		self.flag_placeholder = False

	# build a LSTM network
	def build_cell(self):
		cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_dim, state_is_tuple=True)
		return cell

	def construct_placeholders(self):
		if (self.flag_kernel_opened) :
			print("reset kernel")
			tf.reset_default_graph()
		# Input Place holders
		self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim+self.output_dim])
		self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

		# Build a LSTM network
		multi_cells = tf.contrib.rnn.MultiRNNCell([self.build_cell() for _ in range(self.LSTM_stack)], state_is_tuple=True)
		outputs, _states=tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
		self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_dim)

		# Cost & Loss & Optimizer
		loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		train = optimizer.minimize(loss)

		# RMSE
		self.targets = tf.placeholder(tf.float32, [None, self.output_dim])
		self.predictions = tf.placeholder(tf.float32, [None, self.output_dim])
		self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))
        
		self.flag_kernel_opened = True
		self.flag_placeholder = True

	def restore_network(self, location='checkpoints'):
		if (not self.flag_placeholder) :
			self.construct_placeholders()
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(location))

	def infer(self, data_processor = None, default_ = 0.39):
		if data_processor is None:
			data_processor = self.data_processor
		prediction = []

		# Test step
		testX = data_processor.data
		for idx in range(len(testX)) :
			if idx == 0:
				for j in range(data_processor.label_dim):
					for l in range(data_processor.seq_length):
						testX[idx, l, j + (data_processor.index_dim + data_processor.data_dim - 1)] = default_
				test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: [testX[idx]]})
			elif idx < data_processor.seq_length:
				for j in range(data_processor.label_dim):
					for l in range(data_processor.seq_length):
						if data_processor.seq_length == l:
							testX[idx, l, j + (data_processor.index_dim + data_processor.data_dim - 1)] = test_predict[0][j]
						elif idx < l:
							testX[idx, l, j + (data_processor.index_dim + data_processor.data_dim - 1)] = testX[idx-(data_processor.seq_length-l),data_processor.seq_length-1,j+data_processor.data_dim]
						else:
							testX[idx, l, j + (data_processor.index_dim + data_processor.data_dim - 1)] = default_
				test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: [testX[idx]]})
			else: 
				for j in range(data_processor.label_dim):
					for l in range(data_processor.seq_length):
						if data_processor.seq_length == l:
							testX[idx, l, j + (data_processor.index_dim + data_processor.data_dim - 1)] = test_predict[0][j]
						elif idx < l:
							testX[idx, l, j + (data_processor.index_dim + data_processor.data_dim - 1)] = testX[idx-(data_processor.seq_length-l),data_processor.seq_length-1,j+data_processor.data_dim]
				test_predict = self.sess.run(self.Y_pred, feed_dict= {self.X: [testX[idx]]})
    
			prediction.append(test_predict[0])
        
        # Calculate RMSE
		rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: data_processor.label, self.predictions: prediction})
		print("RMSE: {}".format(rmse_val))
		return prediction

	def close(self):
		tf.reset_default_graph()
		self.sess.close()