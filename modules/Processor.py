import tensorflow as tf
import Encoder
encoder = Encoder.preprocessor()
import os

"""
Tensorflow LSTM Network for Regression from EMG to FLEX
"""

class network:
	def __init__(self, data_encoder = encoder(), hidden_dim = 30, learning_rate = 0.01, LSTM_stack = 2):
		tf.set_random_seed(777)  # reproducibility
		self.data_encoder = data_encoder
		self.seq_length = data_encoder.seq_length
		self.data_dim = data_encoder.data_dim
		self.output_dim = data_encoder.label_dim
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
		# Input Place holders
		self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim+self.output_dim])
		self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

		# Build a LSTM network
		multi_cells = tf.contrib.rnn.MultiRNNCell([self.build_cell() for _ in range(self.LSTM_stack)], state_is_tuple=True)
		outputs, _states=tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
		self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_dim)

		# Cost & Loss & Optimizer
		self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train = self.optimizer.minimize(self.loss)

		# RMSE
		self.targets = tf.placeholder(tf.float32, [None, self.output_dim])
		self.predictions = tf.placeholder(tf.float32, [None, self.output_dim])
		self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))
        
		self.flag_kernel_opened = True
		self.flag_placeholder = True
        
	def train_network(self, training_data = [], training_label = [], iterations = 5000, location = 'model/temp'):
		self.sess = tf.Session(graph=self.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)

		# Training step
		for i in range(iterations):
			_, step_loss = self.sess.run([self.train, self.loss], feed_dict={self.X: training_data, self.Y: training_label})
			if(i % 1000 == 0) :
				print("[step: {}] loss: {}".format(i, step_loss))
		#train_predict = self.sess.run(self.Y_pred, feed_dict={self.X: training_data})
    
		# Save Network
		self.saver = tf.train.Saver()
		self.saver.save(self.sess, location+"/lstm.ckpt")

	def restore_network(self, location='model/_'):
		if (not self.flag_placeholder) :
			self.construct_placeholders()
		if (os.path.exists(location) == False):
			print("Error : No such location")
			return
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(location))

	def infer(self, data_encoder = None, default_ = 0.39):
		if data_encoder is None: data_encoder = self.data_encoder

		prediction = []
		# Test step
		testX = data_encoder.data
		for idx in range(len(testX)) :
			if idx == 0:
				for j in range(data_encoder.label_dim):
					for l in range(data_encoder.seq_length):
						testX[idx, l, j + (data_encoder.index_dim + data_encoder.data_dim - 1)] = default_
				test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: [testX[idx]]})
			elif idx < data_encoder.seq_length:
				for j in range(data_encoder.label_dim):
					for l in range(data_encoder.seq_length):
						if data_encoder.seq_length == l:
							testX[idx, l, j + (data_encoder.index_dim + data_encoder.data_dim - 1)] = test_predict[0][j]
						elif idx < l:
							testX[idx, l, j + (data_encoder.index_dim + data_encoder.data_dim - 1)] = testX[idx-(data_encoder.seq_length-l),data_encoder.seq_length-1,j+data_encoder.data_dim]
						else:
							testX[idx, l, j + (data_encoder.index_dim + data_encoder.data_dim - 1)] = default_
				test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: [testX[idx]]})
			else: 
				for j in range(data_encoder.label_dim):
					for l in range(data_encoder.seq_length):
						if data_encoder.seq_length == l:
							testX[idx, l, j + (data_encoder.index_dim + data_encoder.data_dim - 1)] = test_predict[0][j]
						elif idx < l:
							testX[idx, l, j + (data_encoder.index_dim + data_encoder.data_dim - 1)] = testX[idx-(data_encoder.seq_length-l),data_encoder.seq_length-1,j+data_encoder.data_dim]
				test_predict = self.sess.run(self.Y_pred, feed_dict= {self.X: [testX[idx]]})
    
			prediction.append(test_predict[0])
        
        # Calculate RMSE
		rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: data_encoder.label, self.predictions: prediction})
		print("RMSE: {}".format(rmse_val))
		return prediction

	def close(self):
		tf.reset_default_graph()
		self.sess.close()