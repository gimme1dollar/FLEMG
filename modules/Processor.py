import tensorflow as tf
import Encoder
encoder = Encoder.encoder()
import os

"""
Tensorflow LSTM Network for Regression from EMG to FLEX
"""
class preprocessor:
	def __init__(self, encoder = encoder(), encoded_data = [], data_dim = 8, label_dim = 6, index_dim = 1, seq_length = 3):
		self.encoded_data = encoded_data
        
		self.processed_data = []
		self.label = []
		self.index = []
        
		self.data_dim = encoder.emg_dim + encoder.flex_dim
		self.label_dim = encoder.flex_dim
		self.index_dim = encoder.index_dim
		self.seq_length = seq_length

	def load(self, location='default', delimiter = ','):
		self.encoded_data = np.loadtxt(location, delimiter = delimiter)
        
	def save(self, location='default', delimiter = ','):
    # TODO
		pass
        
	def scale(self, emg_max = 1024, flex_max = 128, chunk = 30):
		w_e=emg_max/chunk
		w_f=flex_max/chunk
		e_f=emg_max/flex_max

		denominator = [1/w_e, e_f,e_f,e_f,e_f,e_f,e_f,e_f,e_f,chunk,chunk,chunk,chunk,chunk,chunk]
		self.encoded_data = np.round(self.encoded_data/denominator)/(w_e)
        
	def preprocess(self):
		if (self.seq_length <= len(self.data)):
			print("Error : seqence length " + self.seq_length + " is shorter than data count " + len(self.processed_data))
			return
            
		dataX = []
		dataY = []
		dataT = []
		for i in range(len(self.encoded_data) - self.seq_length + 1):
			_x = self.encoded_data[i:i+self.seq_length, self.index_dim:]
			_y = self.encoded_data[i+self.seq_length-1, self.data_dim:]  # Next close price
			_t = self.encoded_data[i:i+self.seq_length, :self.index_dim]
			dataX.append(_x)
			dataY.append(_y)
			dataT.append(_t)
		
		self.index = np.array(dataT)
    self.processed_data = np.array(dataX)
		self.label = np.array(dataY)

    return self.index, self.data, self.label

class network:
	def __init__(self, data_encoder = encoder(), hidden_dim = 30, learning_rate = 0.01, LSTM_stack = 2):
		tf.set_random_seed(777)  # reproducibility
		self.data_encoder = data_encoder
		self.seq_length = data_encoder.seq_length
		self.data_dim = data_encoder.flex_dim +  data_encoder.emg_dim
		self.output_dim = data_encoder.flex_dim
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
		self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
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
        
	def train(self, training_data = [], training_label = [], iterations = 5000, location = 'model/temp'):
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

	def restore(self, location='model/_'):
		if (not self.flag_placeholder) :
			self.construct_placeholders()
		if (os.path.exists(location) == False):
			print("Error : No such location")
			return
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(location))

	def infer(self, testSet = [], default_ = 0.39):
		prediction = []
		for idx in range(len(testSet)) :
			if idx == 0:
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						testX[idx, l, j + (self.data_encoder.index_dim + self.data_encoder.data_dim - 1)] = default_
				test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: [testX[idx]]})
			elif idx < self.data_encoder.seq_length:
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						if self.data_encoder.seq_length == l:
							testX[idx, l, j + (self.data_encoder.index_dim + self.data_encoder.data_dim - 1)] = test_predict[0][j]
						elif idx < l:
							testX[idx, l, j + (self.data_encoder.index_dim + self.data_encoder.data_dim - 1)] = testX[idx-(self.data_encoder.seq_length-l),self.data_encoder.seq_length-1,j+self.data_encoder.data_dim]
						else:
							testX[idx, l, j + (self.data_encoder.index_dim + self.data_encoder.data_dim - 1)] = default_
				test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: [testX[idx]]})
			else: 
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						if self.data_encoder.seq_length == l:
							testX[idx, l, j + (self.data_encoder.index_dim + self.data_encoder.data_dim - 1)] = test_predict[0][j]
						elif idx < l:
							testX[idx, l, j + (self.data_encoder.index_dim + self.data_encoder.data_dim - 1)] = testX[idx-(self.data_encoder.seq_length-l),self.data_encoder.seq_length-1,j+self.data_encoder.data_dim]
				test_predict = self.sess.run(self.Y_pred, feed_dict= {self.X: [testX[idx]]})
    
			prediction.append(test_predict[0])
        
        # Calculate RMSE
		rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: data_encoder.label, self.predictions: prediction})
		print("RMSE: {}".format(rmse_val))
		return prediction

	def close(self):
		tf.reset_default_graph()
		self.sess.close()