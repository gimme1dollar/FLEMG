import numpy as np
import queue

class encoder:
	def __init__(self, queue_list = [], index_dim = 1, flex_dim = 5, emg_dim = 8, seq_length = 3):
		self.queue_list = queue_list
		self.dataSet = []
		self.count = 0
        
		self.emg_dim = emg_dim
		self.flex_dim = flex_dim
		self.index_dim = index_dim
		self.data_dim =  emg_dim + flex_dim
		self.label_dim = flex_dim
		self.seq_length = seq_length

	def encode(self, idx = None):
		print('encode')
		if idx is not None:
			tmp = [idx]
		else:
			tmp = [self.count]

		#FLEX : b'343,387,434,410,413\\\n'
		f_d = []
		f_q = self.queue_list[0].queue
		if( len(f_q) > 1 ) :
			f_str = str( f_q[ len(f_q)-1 ] )
			f_list = f_str.split(",")

			if( len(f_list) == self.flex_dim ) :
				f_d.append( int(f_list[0].split('\'')[1]) )
				for i in range( self.flex_dim-2 ):
					f_d.append( int(f_list[i+1]) )
				f_d.append( int(f_list[ self.flex_dim-1 ].split('\\')[0]) )

		#EMG

		#Store
		if( len(f_d) == self.flex_dim ):
			tmp += f_d
			#print(tmp)

			self.dataSet.append(tmp)
			self.count += 1

		for i in range( len(self.queue_list) ):
			self.queue_list[i].queue.clear()