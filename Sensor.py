""" 
Arm Prosthesis using FLEX + EMG data, thus named as FLEMG
Data is composed of (1) 8-channel emg_data (2) 6-channel flex_data
"""

import numpy as np
import serial
import os
import csv

class FLEMG_Data_Collector:
	def __init__(self, flex_size, emg_size):
		self.data = []
		self.count = 0
		self.data_lock = false # Thread : Constant Background Data Acquisition

		self.flex_channels = flex_size
		self.flex_port = None
		self.flex_port_open = false

		self.emg_channels = emg_size
		self.emg_port = None
		self.emg_port_open = false

	def open_port(self, flex_port_name, emg_port_name):
		self.flex_port = serial.Serial(flex_port_name, 115200, timeout=2)
		self.emg_port = serial.Serial(emg_port_name, 19200, timeout=2)
	
		if self.flex_port.is_open:
			self_flex_port_open = True
		else:
			print('FLEX serial not open')
		if self.emg_port.is_open:
			self_emg_port_open = True
		else:
			print('EMG serial not open')

	def close_port(self):
		if self.flex_port.is_open:
			self.flex_port.close()
			self.flex_port_open = False

		if self.emg_port.is_open:
			self.emg_port.close()
			self.emg_port_open = False
		
	def receive_data(self):
		validation = False
		flex_tmp = []
		emg_tmp = []
		
		# Receive a data through port
		if self.flex_port_open and self.flex_port.inWaiting():
			input_flex=self.flex_port.read(self.flex_channels)						decoded_flex=input_flex.decode('ascii')
			
		if self.emg_port_open and self.emg_port.inWaiting():
			input_emg=self.emg_port.read(self.emg_channels)
			decoded_emg=input_emg.decode('ascii')

		for i in range(len(decoded_flex)):
			if decoded_flex[i]=='n'and stack==4 :
				stack=0
				validation=True
			elif decoded_flex[i]=='n' and stack!=4:
				validation=False
				stack=0
				print('FLEX data receiving error')
			elif decoded_flex[i]==',':
				stack += 1
			else:
				flex_tmp.append(decoded_flex[i])

		##check validation of input emg data and get a line of sensor data
		for i in range(len(decoded_emg)):
			if decoded_emg[i]=='[':
				start_index=i;
				stack=0 
			elif decoded_emg[i]==']' and stack==7 :
				end_index=i;
				stack=0
				validation=True
			elif decoded_emg[i]==']' and stack!=7 :
				validation=False
				stack=0
				print('EMG data receiving error')
			elif decoded_emg[i]==',':
				stack += 1
			else:
				emg_tmp.append(decoded_emg[i])

		# Increment self.count
		if validation is True and data_lock is False:
			data_lock = True
			self.data.append([count] + flex_tmp + emg_tmp)
			self.count = self.count + 1
			data_lock = False

	def scale_data(self, emg_max = 1024, flex_max = 128, chunk = None):
		if chunk is None:
			chunk = 1

		w_e=emg_max/chunk
		w_f=flex_max/chunk
		e_f=emg_max/flex_max
		d  = [1/w_e] + [e_f]*self.emg_channels + [chunk]*self.flex_channels

		While(True):
			if self.data_lock is False:
				self.data_lock = True
				for i in range(count):
					self.data[i] = np.round(self.data[i]/d) 
					self.data[i] = self.data[i]/ w_e
				break
		self.data_lock = False


	def print_data(self, count = None):
		While(True):
			if self.data_lock is False:
				self.data_lock = True
				if count is not None:
					print(self.data[-count:])
				else:
					print(self.data)
				break
			self.data_lock = False

	def build_training_dataset(self, seq_length = 3):
	""" 
	(x,y) -> (y*)  : dataX -> dataY(label)
	"""
		While(True):
			if self.data_lock is False:
				self.data_lock = True
				dataX = []
				dataY = []
				for i in range(count - seq_length):
			        		_x = self.data[i:i + seq_length, :]
					_y = self.data[i + seq_length, -self.flex_channels]
					dataX.append(_x)
					dataY.append(_y)
				break
		self.data_lock = False	
		return np.array(dataX), np.array(dataY)

	def save_data_set(self, location='default'):
		print("Save FLEMG dataset to " + str(location))

	def load_data_set(self, location='default'):
		While(True):
			if self.data_lock is False:
				self.data_lock = True
				if self.count == 0:
					self.data = np.loadtxt(location, delimiter=',')
				else:
					self.data.append(np.loadtxt(location, delimiter=','))
				self.count = int(len(self.data))
				break
		self.data_lock = False
		print("Load FLEMG dataset from " + str(location))

	def enable_thread(self):
		self.data_lock = False
		
	def disable_thread(self):
		self.data_lock = True


