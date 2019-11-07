import numpy as np

class preprocessor:
	def __init__(self, raw = [], data_dim = 8, label_dim = 6, index_dim = 1, seq_length = 3):
		self.raw = raw
        
		self.data = []
		self.label = []
		self.count = 0
		self.index = []
        
		self.data_dim = data_dim
		self.label_dim = label_dim
		self.index_dim = index_dim
		self.seq_length = seq_length

	def load(self, location='default', delimiter = ','):
		self.raw = np.loadtxt(location, delimiter = delimiter)
        
	def scale(self, emg_max = 1024, flex_max = 128, chunk = 30):
		w_e=emg_max/chunk
		w_f=flex_max/chunk
		e_f=emg_max/flex_max

		denominator = [1/w_e, e_f,e_f,e_f,e_f,e_f,e_f,e_f,e_f,chunk,chunk,chunk,chunk,chunk,chunk]
		self.raw = np.round(self.raw/denominator)/(w_e)
        
	def preprocess(self):
		if (self.seq_length <= len(self.data)):
			print("Error : seqence length " + self.seq_length + " is shorter than data count " + len(self.data))
			return
            
		dataX = []
		dataY = []
		dataT = []
		for i in range(len(self.raw) - self.seq_length + 1):
			_x = self.raw[i:i+self.seq_length, self.index_dim:]
			_y = self.raw[i+self.seq_length-1, self.index_dim+self.data_dim:]  # Next close price
			_t = self.raw[i:i+self.seq_length, :self.index_dim]
			dataX.append(_x)
			dataY.append(_y)
			dataT.append(_t)
		self.data = np.array(dataX)
		self.label = np.array(dataY)
		self.count = len(self.data)
		self.index = np.array(dataT)

