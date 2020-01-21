import tensorflow as tf
import numpy as np
import os
try:
	from _modules import Encoder
except ImportError:
	import Encoder
import random

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
		print("Save Start")
		whole = np.concatenate((self.index[0][0] , self.data[0][0]), axis=0)
		for i in range( len(self.index)-1 ):
			tmp = np.concatenate((self.index[i+1][0] , self.data[i+1][0]), axis=0)
			whole = np.vstack((whole, tmp))
		for s in range( self.encoder.seq_length-1 ):
			whole = np.vstack((whole, np.concatenate(( self.index[i+1][s+1] , self.data[i+1][s+1] ), axis = 0)))

		np.savetxt(location, whole, fmt='%s', delimiter = delimiter)
		print("Save End")

	def scale(self, emg_max = 187500.016, flex_max = 1024):
		#w_e=emg_max/chunk
		#e_f=emg_max/flex_max
		#denominator = [1/w_e, e_f,e_f,e_f,e_f,e_f,e_f,e_f,e_f,chunk,chunk,chunk,chunk,chunk,chunk]
		#self.raw = np.round(self.raw/denominator)/(w_e)
		self.raw[:,self.encoder.index_dim:self.encoder.index_dim+self.encoder.emg_dim]/=emg_max
		self.raw[:,self.encoder.index_dim:self.encoder.index_dim+self.encoder.emg_dim]=(self.raw[:,self.encoder.index_dim:self.encoder.index_dim+self.encoder.emg_dim]--0.08)/(0.08--0.08)
		self.raw[:,self.encoder.index_dim+self.encoder.emg_dim:]/=flex_max
		self.raw[:,self.encoder.index_dim+self.encoder.emg_dim:]=(self.raw[:,self.encoder.index_dim+self.encoder.emg_dim:]-0.15)/(0.4-0.15)
		#denominator = [1,emg_max,emg_max,emg_max,emg_max,emg_max,emg_max,emg_max,emg_max,flex_max,flex_max,flex_max,flex_max,flex_max]
		#self.raw = self.raw/denominator

	def preprocess_default(self, data = None):
		if data is not None:
			self.raw = np.asarray(data)

		if (self.encoder.seq_length >= len(self.raw)):
			print(f"Error : seqence length {self.encoder.seq_length} is shorter than data count {len(self.raw)}")
			return

		dataX = []
		dataY = []
		dataT = []
		for i in range(len(self.raw) - self.encoder.seq_length + 1):
			_x = self.raw[i:i+self.encoder.seq_length, self.encoder.index_dim : self.encoder.index_dim+self.encoder.emg_dim] # Input이 emg데이터 부분만
			_y = self.raw[i+self.encoder.seq_length-1, self.encoder.data_dim+self.encoder.index_dim-self.encoder.label_dim:] 
			_t = self.raw[i:i+self.encoder.seq_length, :self.encoder.index_dim]
			dataX.append(_x)
			dataY.append(_y)
			dataT.append(_t)
		
		self.index = np.array(dataT)
		self.data = np.array(dataX)
		self.label = np.array(dataY)

		return self.index, self.data, self.label   
  
	def preprocess_feedback_flex(self, data = None):
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
			_y = self.raw[i+self.encoder.seq_length-1, self.encoder.data_dim+self.encoder.index_dim-self.encoder.label_dim:] 
			_t = self.raw[i:i+self.encoder.seq_length, :self.encoder.index_dim]
			dataX.append(_x)
			dataY.append(_y)
			dataT.append(_t)
		
		self.index = np.array(dataT)
		self.data = np.array(dataX)
		self.label = np.array(dataY)

		return self.index, self.data, self.label

	def preprocess_feedback_whole(self, data = None):
		if data is not None:
			self.raw = np.asarray(data)

		if (self.encoder.seq_length >= len(self.raw)):
			print(f"Error : seqence length {self.encoder.seq_length} is shorter than data count {len(self.raw)}")
			return

		dataX = []
		dataY = []
		dataT = []
		for i in range(len(self.raw) - self.encoder.seq_length + 1):
			_x = self.raw[i:i+self.encoder.seq_length, self.encoder.index_dim:self.encoder.index_dim+self.encoder.emg_dim]
			label = self.raw[i:i + self.encoder.seq_length, self.encoder.data_dim+self.encoder.index_dim-self.encoder.label_dim:]
			_x = np.concatenate( (np.concatenate((_x, _x), axis=1), label), axis=1 )
			_y = self.raw[i+self.encoder.seq_length-1, self.encoder.index_dim:] # Label이 emg+flex데이터가 되도록
			_t = self.raw[i:i+self.encoder.seq_length, :self.encoder.index_dim]
			dataX.append(_x)
			dataY.append(_y)
			dataT.append(_t)
		
		self.index = np.array(dataT)
		self.data = np.array(dataX)
		self.label = np.array(dataY)

		return self.index, self.data, self.label

	def preprocess_flex(self, data=None):
		if data is not None:
			self.raw = np.asarray(data)

		if (self.encoder.seq_length >= len(self.raw)):
			print(f"Error : seqence length {self.encoder.seq_length} is shorter than data count {len(self.raw)}")
			return

		dataX = []
		dataY = []
		dataT = []
		for i in range(len(self.raw) - self.encoder.seq_length + 1):
			_x = self.raw[i:i+self.encoder.seq_length, self.encoder.data_dim+self.encoder.index_dim-self.encoder.label_dim:]
			_y = self.raw[i+self.encoder.seq_length-1, self.encoder.data_dim+self.encoder.index_dim-self.encoder.label_dim:]
			_t = self.raw[i:i+self.encoder.seq_length, :self.encoder.index_dim]
			dataX.append(_x)
			dataY.append(_y)
			dataT.append(_t)

		self.index = np.array(dataT)
		self.data = np.array(dataX)
		self.label = np.array(dataY)

		return self.index, self.data, self.label

	def preprocess_feature_average(self, data = None):
		self.data_dim = self.encoder.emg_dim*self.encoder.seq_length+self.encoder.flex_dim
		print(f"data_dim : {self.data_dim}")
		if data is not None:
			self.raw = np.asarray(data)

		if (self.encoder.seq_length >= len(self.raw)):
			print(f"Error : seqence length {self.encoder.seq_length} is shorter than data count {len(self.raw)}")
			return

		dataX = []
		dataY = []
		dataT = []
		for i in range(len(self.raw) - self.encoder.seq_length + 1):
			_x = self.raw[i:i+self.encoder.seq_length, self.encoder.index_dim : self.encoder.emg_dim+1] # Input이 emg데이터 부분만
			_y = self.raw[i+self.encoder.seq_length-1, self.data_dim-self.encoder.label_dim+1:] 
			_t = self.raw[i:i+self.encoder.seq_length, :self.encoder.index_dim]
			dataX.append(_x)
			dataY.append(_y)
			dataT.append(_t)
		
		self.index = np.array(dataT)
		self.data = np.array(dataX)
		self.label = np.array(dataY)

		return self.index, self.data, self.label   

	def feature_average(self, average_unit1 = 25, average_unit2 = 250, data = None):
		if data is not None:
			self.raw = np.asarray(data)

		res = []
		for i in range(len(self.raw)):
			if i > average_unit2:
				tmp = [self.raw[i, :self.encoder.index_dim]]
				for ch in range(self.encoder.emg_dim):
					tmp.append( self.raw[i, self.encoder.index_dim+ch] )
					
				for ch in range(self.encoder.emg_dim):
					avg1 = 0
					for r in range(average_unit1):
						avg1 += self.raw[i-r, self.encoder.index_dim+ch]
					tmp.append( avg1/average_unit1 )

				for ch in range(self.encoder.emg_dim):
					avg2 = 0
					for r in range(average_unit2):
						avg2 += self.raw[i-r, self.encoder.index_dim+ch]
					tmp.append( avg2/average_unit2 )
						
				tmp = np.concatenate( (tmp, self.raw[i, 1+self.encoder.emg_dim:]), axis=None)
					
				res.append(tmp.tolist())
		
		print(res[0])
		return res

class network_default:
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

	def construct_placeholders(self, learning_rate = 0.1, hidden_dim = 30, stack_dim = 2):
		self.learning_rate = learning_rate
		self.hidden_dim = hidden_dim
		self.stack_dim = stack_dim

		# Input Place holders
		self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim-self.output_dim])
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
		self.rmse=[]
		for i in range(self.output_dim):
			rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets[:,i] - self.predictions[:,i])))
			self.rmse.append(rmse)
		#self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))
        
		self.flag_kernel_opened = True
		self.flag_placeholder = True
        
	def train_network(self, training_data = [], training_label = [], iterations = 5000, location = 'model/_'):
		self.sess = tf.Session(graph=self.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.loss_set=[]

		# Training step
		for i in range(iterations):
			_, step_loss = self.sess.run([self.train, self.loss], feed_dict={self.X: training_data, self.Y: training_label})
			self.loss_set.append(step_loss)
			if(i) :
				print(f"[step: {i}] loss: {step_loss}")
		#train_predict = self.sess.run(self.Y_pred, feed_dict={self.X: training_data})

		# Save Network
		self.saver = tf.train.Saver()
		self.saver.save(self.sess, location+"/lstm.ckpt")
		return self.loss_set

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
		for idx in range(len(testSet)) :
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

class network_feedback_flex:
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

	def construct_placeholders(self, learning_rate = 0.1, hidden_dim = 30, stack_dim = 2):
		self.learning_rate = learning_rate
		self.hidden_dim = hidden_dim
		self.stack_dim = stack_dim

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
		self.rmse=[]
		for i in range(self.output_dim):
			rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets[:,i] - self.predictions[:,i])))
			self.rmse.append(rmse)
		#self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))
        
		self.flag_kernel_opened = True
		self.flag_placeholder = True
        
	def train_network(self, training_data = [], training_label = [], iterations = 5000, location = 'model/_'):
		self.sess = tf.Session(graph=self.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.loss_set=[]
		# Training step
		for i in range(iterations):
			_, step_loss = self.sess.run([self.train, self.loss], feed_dict={self.X: training_data, self.Y: training_label})
			self.loss_set.append(step_loss)
			if(i % 1 == 0) :
				print(f"[step: {i}] loss: {step_loss}")
		#train_predict = self.sess.run(self.Y_pred, feed_dict={self.X: training_data})
    
		# Save Network
		self.saver = tf.train.Saver()
		#location = location + "(" + str(self.stack_dim) + ")"
		self.saver.save(self.sess, location+"/lstm.ckpt")
		return self.loss_set

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
						#testSet[idx, l, self.data_encoder.emg_dim + j] = random.random()
			else:
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						if l == self.data_encoder.seq_length - 1:
							testSet[idx, l, self.data_encoder.emg_dim + j] = test_predict[0][j]
						else:
							testSet[idx, l, self.data_encoder.emg_dim + j] = testSet[idx-1, l+1, self.data_encoder.emg_dim + j]

			# Feed testData
			#print(f"{idx} test\n{testSet[idx,:,8:]}\n")
			test_predict = self.sess.run(self.Y_pred, feed_dict= {self.X: [testSet[idx]]})
			#print(f"{idx} pred\n{test_predict[0]}\n\n")
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

class network_feedback_all:
	def __init__(self, data_encoder = encoder()):
		tf.set_random_seed(777)  # reproducibility
		self.data_encoder = data_encoder
		self.seq_length = data_encoder.seq_length
		self.data_dim = data_encoder.flex_dim +  data_encoder.emg_dim*2
		self.output_dim = data_encoder.flex_dim + data_encoder.emg_dim
        
		self.graph = tf.Graph()
		self.sess = tf.Session()
        
		self.flag_kernel_opened = False
		self.flag_placeholder = False

	# build a LSTM network
	def build_cell(self):
		cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_dim, state_is_tuple=True)
		return cell

	def construct_placeholders(self, learning_rate = 0.1, hidden_dim = 30, stack_dim = 2):
		self.learning_rate = learning_rate
		self.hidden_dim = hidden_dim
		self.stack_dim = stack_dim

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
		self.rmse=[]
		for i in range(self.output_dim):
			rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets[:,i] - self.predictions[:,i])))
			self.rmse.append(rmse)
		#self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))
        
		self.flag_kernel_opened = True
		self.flag_placeholder = True
        
	def train_network(self, training_data = [], training_label = [], iterations = 5000, location = 'model/_'):
		self.sess = tf.Session(graph=self.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.loss_set=[]

		# Training step
		for i in range(iterations):
			_, step_loss = self.sess.run([self.train, self.loss], feed_dict={self.X: training_data, self.Y: training_label})
			self.loss_set.append(step_loss)
			if(i) :
				print(f"[step: {i}] loss: {step_loss}")
		
		#train_predict = self.sess.run(self.Y_pred, feed_dict={self.X: training_data})

		# Save Network
		self.saver = tf.train.Saver()
		#location = location + "(" + str(self.stack_dim) + ")"
		self.saver.save(self.sess, location+"/lstm.ckpt")
		return self.loss_set

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
				for j in range(self.data_encoder.emg_dim + self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						testSet[idx, l, j] = default_
						#testSet[idx, l, self.data_encoder.emg_dim + j] = random.random()
			else:
				for j in range(self.data_encoder.emg_dim + self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						if l == self.data_encoder.seq_length - 1:
							testSet[idx, l, j] = test_predict[0][j]
						else:
							testSet[idx, l, j] = testSet[idx-1, l+1, j]

			# Feed testData
			test_predict = self.sess.run(self.Y_pred, feed_dict= {self.X: [testSet[idx]]})
			prediction.append(test_predict[0])

        # Calculate RMSE
		if testLabel is not None:
			rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: testLabel, self.predictions: prediction})
			#print(f"RMSE: {rmse_val}\n")
			return prediction, rmse_val
		return prediction

	def close(self):
		tf.reset_default_graph()
		self.sess.close()

class network_feature_average:
	def __init__(self, data_encoder = encoder()):
		tf.set_random_seed(777)  # reproducibility
		self.data_encoder = data_encoder
		self.seq_length = data_encoder.seq_length
		self.data_dim = data_encoder.flex_dim + (data_encoder.emg_dim * 3)
		self.output_dim = data_encoder.flex_dim
        
		self.graph = tf.Graph()
		self.sess = tf.Session()
        
		self.flag_kernel_opened = False
		self.flag_placeholder = False

	# build a LSTM network
	def build_cell(self):
		cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_dim, state_is_tuple=True)
		return cell

	def construct_placeholders(self, learning_rate = 0.1, hidden_dim = 30, stack_dim = 2):
		self.learning_rate = learning_rate
		self.hidden_dim = hidden_dim
		self.stack_dim = stack_dim

		# Input Place holders
		self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_encoder.emg_dim])
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
		self.rmse=[]
		for i in range(self.output_dim):
			rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets[:,i] - self.predictions[:,i])))
			self.rmse.append(rmse)
		#self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))
        
		self.flag_kernel_opened = True
		self.flag_placeholder = True
        
	def train_network(self, training_data = [], training_label = [], iterations = 5000, location = 'model/_'):
		self.sess = tf.Session(graph=self.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.loss_set=[]
		# Training step
		for i in range(iterations):
			_, step_loss = self.sess.run([self.train, self.loss], feed_dict={self.X: training_data, self.Y: training_label})
			self.loss_set.append(step_loss)
			if(i) :
				print(f"[step: {i}] loss: {step_loss}")
		#train_predict = self.sess.run(self.Y_pred, feed_dict={self.X: training_data})

		# Save Network
		self.saver = tf.train.Saver()
		#location = location + "(" + str(self.stack_dim) + ")"
		self.saver.save(self.sess, location+"/lstm.ckpt")
		return self.loss_set

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
		for idx in range(len(testSet)) :
			test_predict = self.sess.run(self.Y_pred, feed_dict= {self.X: [testSet[idx]]})
			prediction.append(test_predict[0])

        # Calculate RMSE
		if testLabel is not None:
			rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: testLabel, self.predictions: prediction})
			print(f"RMSE: {rmse_val}\n")
			return prediction, rmse_val
		return prediction

        # Calculate RMSE
		if testLabel is not None:
			rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: testLabel, self.predictions: prediction})
			print(f"RMSE: {rmse_val}\n")
			return prediction, rmse_val
		return prediction

	def close(self):
		tf.reset_default_graph()
		self.sess.close()


class network_flex:
	def __init__(self, data_encoder=encoder()):
		tf.set_random_seed(777)  # reproducibility
		self.data_encoder = data_encoder
		self.seq_length = data_encoder.seq_length
		self.data_dim = data_encoder.flex_dim
		self.output_dim = data_encoder.flex_dim

		self.graph = tf.Graph()
		self.sess = tf.Session()

		self.flag_kernel_opened = False
		self.flag_placeholder = False

	# build a LSTM network
	def build_cell(self):
		cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim, state_is_tuple=True)
		return cell

	def construct_placeholders(self, learning_rate=0.1, hidden_dim=30, stack_dim=2):
		self.learning_rate = learning_rate
		self.hidden_dim = hidden_dim
		self.stack_dim = stack_dim

		# Input Place holders
		self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
		self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

		# Build a LSTM network
		multi_cells = tf.contrib.rnn.MultiRNNCell([self.build_cell() for _ in range(self.stack_dim)],
												  state_is_tuple=True)
		outputs, _states = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
		self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_dim)

		# Cost & Loss & Optimizer
		self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train = self.optimizer.minimize(self.loss)

		# RMSE
		self.targets = tf.placeholder(tf.float32, [None, self.output_dim])
		self.predictions = tf.placeholder(tf.float32, [None, self.output_dim])
		self.rmse = []
		for i in range(self.output_dim):
			rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets[:, i] - self.predictions[:, i])))
			self.rmse.append(rmse)
		# self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

		self.flag_kernel_opened = True
		self.flag_placeholder = True

	def train_network(self, training_data=[], training_label=[], iterations=5000, location='model/_'):
		self.sess = tf.Session(graph=self.graph)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.loss_set = []

		# Training step
		for i in range(iterations):
			_, step_loss = self.sess.run([self.train, self.loss],
										 feed_dict={self.X: training_data, self.Y: training_label})
			self.loss_set.append(step_loss)
			if (i):
				print(f"[step: {i}] loss: {step_loss}")
		# train_predict = self.sess.run(self.Y_pred, feed_dict={self.X: training_data})

		# Save Network
		self.saver = tf.train.Saver()
		# location = location + "(" + str(self.stack_dim) + ")"
		self.saver.save(self.sess, location + "/lstm.ckpt")
		return self.loss_set

	def restore(self, location='model/_'):
		if (not self.flag_placeholder):
			self.construct_placeholders()
		if (os.path.exists(location) == False):
			print("Error : No such location")
			return
		self.sess = tf.Session(graph=self.graph)

		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(location))

	def infer(self, testSet=[], testLabel=None, default_=0.39):
		prediction = []
		for idx in range(len(testSet)):
			# testSet Reconstruction
			if idx == 0:
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						testSet[idx, l, j] = default_
			else:
				for j in range(self.data_encoder.label_dim):
					for l in range(self.data_encoder.seq_length):
						if l == self.data_encoder.seq_length - 1:
							testSet[idx, l, j] = test_predict[0][j]
						else:
							testSet[idx, l, j] = testSet[idx-1, l+1, j]

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