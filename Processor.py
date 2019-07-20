import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

"""
Tensorflow LSTM Network for Regression from EMG to FLEX
***Yet Editing***
"""

class LSTM_Network:
	def __init__(self, seq_length = 3, data_dim = 8, hidden_dim = 30, output_dim = 6, learning_rate = 0.01, LSTM_stack = 2):
		tf.set_random_seed(777)  # reproducibility
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

		for idx in range(len(testX)) :
			for j in range(6):
				testX[idx, 0, j+8] = testY[idx,j]
				testX[idx, 1, j+8] = testY[idx,j]
				testX[idx, 2, j+8] = testY[idx,j]
                    
		test_predict = self.sess.run(Y_pred, feed_dict= {X: [testX[idx]]})
		prediction.append(test_predict[0]) 
            
		# RMSE
		rmse_val = self.sess.run(rmse, feed_dict={targets: testY, predictions: prediction})
		print("RMSE: {}".format(rmse_val))

	def close():
		self.session.close()
