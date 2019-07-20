""" 
Data is composed of (1) time_stamp (2) 8-channel emg_data (3) 6-channel flex_data

"""

import numpy as np
import serial
import os

class FLEMG_Data_Collector:
	def __init__(self, flex_size, emg_size):
		self.data = []
		self.count = 0

		self.flex_data_size = flex_size
		self.fleg_port_flag = false

		self.emg_data_size = emg_size
		self.emg_port_flag = false
	
	def open_port(self, flex_port_name, emg_port_name):
		self.flex_port = serial.Serial(flex_port_name, 115200, timeout=2)
		self.emg_port = serial.Serial(emg_port_name, 19200, timeout=2)
	
		if self.flex_port.is_open:
			self_flex_port_flag = True
		else:
			print('FLEX serial not open')
		if self.emg_port.is_open:
			self_emg_port_flag = True
		else:
			print('EMG serial not open')

	def close_port(self):
		if self.flex_port.is_open:
			self.flex_port.close()
			self.flex_port_flag = False

		if self.emg_port.is_open:
			self.emg_port.close()
			self.emg_port_flag = False
		
	def receive_data(self):
		validation = False
		flex_tmp = []
		emg_tmp = []
		
		# Receive a data through port
		if self.flex_port_flag and self.flex_port.inWaiting():
			input_flex=self.flex_port.read(self.flex_data_size)						decoded_flex=input_flex.decode('ascii')
			
		if self.emg_port_flag and self.emg_port.inWaiting():
			input_emg=self.emg_port.read(self.emg_data_size)
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
		if validation is True:
			self.data.append([count] + flex_tmp + emg_tmp)
			self.count = self.count + 1

	def print_data(self, count = None):
		if count is not None:
			print(self.data[-count:])
		else:
			print(self.data)
		

	def save_data_set(self, location='default'):
		print("Save FLEMG dataset to " + str(location))

	def load_data_set(self, location='default'):
		print("Load FLEMG dataset from " + str(location))

