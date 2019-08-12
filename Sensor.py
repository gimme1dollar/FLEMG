import numpy as np
import serial
import os
import time

class sensor:
	def __init__(self, flex_size = 6, emg_size = 8):
		self.data = []
		self.count = 0
		self.init_time = int(time.time() * 1000)

		self.flex_data_size = flex_size
		self.flex_port = None
		self.fleg_port_flag = False

		self.emg_data_size = emg_size
		self.emg_port = None
		self.emg_port_flag = False
    

	def open_port(self, flex_port_name = 'COM17', emg_port_name = 'COM2'):
    # Instantiate and Open Serial Ports
		self.flex_port = serial.Serial(flex_port_name, 115200,timeout=0.004)
		self.emg_port = serial.Serial(emg_port_name, 115200,timeout=0.004)

    # Set member values related to the ports
		if self.flex_port.is_open:
			self_flex_port_flag = True
			print('FLEX serial port open')
		else:
			print('FLEX serial not open')

		if self.emg_port.is_open:
			self_emg_port_flag = True
			print('EMG serial port open')
		else:
			print('EMG serial not open')

	def close_port(self):
		if self.flex_port.is_open:
			self.flex_port.close()
			self.flex_port_flag = False
			print('EMG serial port close')

		if self.emg_port.is_open:
			self.emg_port.close()
			self.emg_port_flag = False
			print('FLEX serial port close')

	def receive_data(self):
		if self.self_emg_port_flag and self.flex_port_flag:
			emg_size = self.emg_port.inWaiting()
			flex_size = self.flex_port.inWaiting()
			emg_number = 12 * self.emg_data_size - 1 # 상수 의미?
			flex_number = 4 * self.flex_data_size - 1 # 상수 의미?

			if (emg_size >= emg_number) and (flex_size >= flex_number):
				input_emg_dataSet = self.emg_port.read(emg_size)
				input_flex_dataSet = self.flex_port.read(flex_size)
				decoded_emg_dataSet = list(input_emg_dataSet.decode('ascii'))
				decoded_flex_dataSet = list(input_flex_dataSet.decode('ascii'))
				decoded_emg_size = len(decoded_emg_dataSet)
				decoded_flex_size = len(decoded_flex_dataSet)

			# Synchronize FLEX and EMG Data
			while decoded_emg_size>=emg_number and decoded_flex_size>=flex_number:
				# Index
				current_time = int(time.time()*1000)
				time_index = current_time - self.initial_time

				# EMG Data
				start_index=0
				end_index=0
				pivot_index=0
				while True:
					if decoded_emg_dataSet[pivot_index]=='[':
						start_index = pivot_index + 1
					elif decoded_emg_dataSet[pivot_index]==']':
						if (pivot_index - start_index==emg_number):
							end_index = pivot_index
							emg_data = decoded_emg_dataSet[start_index:end_index]
							del decoded_emg_dataSet[:end_index+1]
							break
						else:
							del decoded_emg_dataSet[:pivot_index+1]
							pivot_index = -1
					pivot_index = pivot_index + 1
					if (pivot_index >= decoded_emg_size):
						pivot_index=0
						break

				# FLEX Data
				start_index=0
				end_index=0
				pivot_index=0
				while True:
					if decoded_flex_dataSet[pivot_index]=='\n':
						if (pivot_index - start_index == flex_number):
							end_index = pivot_index
							flex_data = decoded_flex_dataSet[start_index:end_index]
							del decoded_flex_dataSet[start_index:end_index+1]
							break
						else:
							del decoded_flex_dataSet[:pivot_index+1]
							pivot_index = -1
					pivot_index = pivot_index + 1
					if (pivot_index >= decoded_flex_size):
						pivot_index = 0
						break
          
				# Stack Data
				temp = [int(time_index)] + emg_data + flex_data
				self.data.append(tmp)
				self.count = self.count + 1
		else:
			print('Serial Not Open')

	def print_data(self, count = None):
		if count is not None:
			print(self.data[-count:])
		else:
			print(self.data)

	def clear_data(self):
		self.data = []
		self.count = 0
            
	def save_data_set(self, location='default.csv', delimiter=',', format ='%d'):
		np.savetxt(location, self.data, delimiter=delimiter, fmt=format)
		print("Save FLEMG dataset to " + str(location))

	def load_data_set(self, location='default', delimiter=','):
		self.data = np.loadtxt(location, delimiter = delimiter)
		self.data_count = len(self.data)
		print("Load FLEMG dataset from " + str(location))