import numpy as np
import queue

class encoder:
	def __init__(self, raw = queue.Queue(), index_dim = 1, flex_dim = 6, emg_dim = 8, seq_length = 3):
		self.raw = raw
		self.dataSet = []
		self.label = []
		self.count = 0
		self.index = []
        
		self.emg_dim = emg_dim
		self.flex_dim = flex_dim
		self.index_dim = index_dim
		self.data_dim =  emg_dim + flex_dim
		self.label_dim = flex_dim
		self.seq_length = seq_length

	def encode(self):
		self.count += 1