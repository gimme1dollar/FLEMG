import tensorflow as tf
import numpy as np
import os
try:
	import Encoder
except ImportError:
	from modules import Encoder

encoder = Encoder.encoder
"""
Tensorflow LSTM Network for Regression from EMG to FLEX
"""
class preprocessor:
	def __init__(self, en = encoder(), raw = []):
		self.raw = raw
		self.encoder = en
		self.data = []
		self.label = []
		self.index = []

	def load(self, location='default', delimiter = ','):
		self.raw = np.loadtxt(location, delimiter = delimiter)
        
	def save(self, location='default', delimiter = ','):
		print(self.index)
		print(self.data)
		whole = np.concatenate((self.index[0][0] , self.data[0][0]), axis=0)
		for i in range( len(self.index)-1 ):
			tmp = np.concatenate((self.index[i+1][0] , self.data[i+1][0]), axis=0)
			whole = np.vstack((whole, tmp))
		for s in range( self.encoder.seq_length-1 ):
			whole = np.vstack((whole, np.concatenate(( self.index[i+1][s+1] , self.data[i+1][s+1] ), axis = 0)))
		np.savetxt(location, whole.astype(int), fmt='%i', delimiter = delimiter)

	def scale(self, emg_max = 1024, flex_max = 128, chunk = 30):
		w_e=emg_max/chunk
		e_f=emg_max/flex_max

		denominator = [1/w_e, e_f,e_f,e_f,e_f,e_f,e_f,e_f,e_f,chunk,chunk,chunk,chunk,chunk,chunk]
		self.raw = np.round(self.raw/denominator)/(w_e)
        
	def preprocess(self, data = None):
		if data is not None:
			self.raw = np.asarray(data)

		if (self.encoder.seq_length >= len(self.raw)):
			print(f"Error : seqence length {self.encoder.seq_length} is shorter than data count {len(self.raw)}")
			return

		dataX = []
		dataY = []
		dataT = []
		for i in range(len(self.raw) - self.encoder.seq_length + 1):
			_x = self.raw[i:i+self.encoder.seq_length, self.encoder.index_dim:]
			_y = self.raw[i+self.encoder.seq_length-1, self.encoder.data_dim-self.encoder.label_dim+1:]  # Next close price
			_t = self.raw[i:i+self.encoder.seq_length, :self.encoder.index_dim]
			dataX.append(_x)
			dataY.append(_y)
			dataT.append(_t)
		
		self.index = np.array(dataT)
		self.data = np.array(dataX)
		self.label = np.array(dataY)

		return self.index, self.data, self.label

class network:
	def __init__(self, data_encoder = encoder()):
		tf.set_random_seed(777)  # reproducibility
		self.data_encoder = data_encoder
		self.seq_length = data_encoder.seq_length
		self.data_dim = data_encoder.flex_dim +  data_encoder.emg_dim
		self.output_dim = data_encoder.flex_dim
        
		self.graph = tf.Graph()
		self.sess = tf.Session()
        
		self.flag_kernel_opened = False
		self.flag_placeholder = False

	# build a LSTM network
	def build_cell(self):
		cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_dim, state_is_tuple=True)
		return cell

	def construct_placeholders(self, learning_rate = 0.1, hidden_dim = 30, stak_dim = 2):
		self.learning_rate = learning_rate
		self.hidden_dim = hidden_dim
		self.stack_dim = stak_dim

		# Input Place holders
		self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
		self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

		# Build a LSTM network
		multi_cells = tf.contrib.rnn.MultiRNNCell([self.build_cell() for _ in range(self.stack_dim)], state_is_tuple=True)
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
        
	def train_network(self, training_data = [], training_label = [], iterations = 5000, location = 'model/_'):
		self.sess = tf.Session(graph=self.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)

		# Training step
		for i in range(iterations):
			_, step_loss = self.sess.run([self.train, self.loss], feed_dict={self.X: training_data, self.Y: training_label})
			if(i % 1000 == 0) :
				print(f"[step: {i}] loss: {step_loss}")
		#train_predict = self.sess.run(self.Y_pred, feed_dict={self.X: training_data})
    
		# Save Network
		self.saver = tf.train.Saver()
		location = location + "(" + str(self.stack_dim) + ")"
		self.saver.save(self.sess, location+"/lstm.ckpt")

	def restore(self, location='model/_'):
		if (not self.flag_placeholder) :
			self.construct_placeholders()
		if (os.path.exists(location) == False):
			print("Error : No such location")
			return
		self.sess = tf.Session(graph=self.graph)

		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(location))

	def infer(self, testSet = [], testLabel = None, default_ = 0.39):
		prediction = []
		for idx in range( len(testSet) ):
			# testSet Reconstruction
			if idx == 0:
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						testSet[idx, l, self.data_encoder.emg_dim + j] = default_
			elif idx < self.data_encoder.seq_length:
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						if l == self.data_encoder.seq_length:
							testSet[idx, l, self.data_encoder.emg_dim + j] = test_predict[0][j]
						elif l > idx:
							testSet[idx, l, self.data_encoder.emg_dim + j] = testSet[idx-(self.data_encoder.seq_length-l), self.data_encoder.seq_length - 1, self.data_encoder.emg_dim + j]
						else:
							testSet[idx, l, self.data_encoder.emg_dim + j] = default_
			else:
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						if l == self.data_encoder.seq_length:
							testSet[idx, l, self.data_encoder.emg_dim + j] = test_predict[0][j]
						elif l > idx:
							testSet[idx, l, self.data_encoder.emg_dim + j] = testSet[idx-(self.data_encoder.seq_length-l), self.data_encoder.seq_length - 1, self.data_encoder.emg_dim + j]
			# Feed testData
			test_predict = self.sess.run(self.Y_pred, feed_dict= {self.X: [testSet[idx]]})
			prediction.append(test_predict[0])

        # Calculate RMSE
		if testLabel is not None:
			rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: testLabel, self.predictions: prediction})
			print(f"RMSE: {rmse_val}\n")
			return prediction, rmse_val
		return prediction

	def close(self):
		tf.reset_default_graph()
		self.sess.close()