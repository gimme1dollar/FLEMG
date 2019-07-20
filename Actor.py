import serial
import os
from Sensor import *

with serial.Serial('COM2', 115200,timeout=2) as emg_ser, serial.Serial('COM15', 19200,timeout=2) as flex_ser:
	if emg_ser.is_open and flex_ser.is_open:
		while True:
			emg_size = emg_ser.inWaiting()
			flex_size = flex_ser.inWaiting()
			if emg_size and flex_size:
				input_emg_dataSet=emg_ser.read(emg_size)
				input_flex_dataSet=flex_ser.read(flex_size)
				print('flex')
				print(input_flex_dataSet)

				decoded_emg_dataSet=input_emg_dataSet.decode('ascii')
				decoded_flex_dataSet=input_flex_dataSet.decode('ascii')
				print('decoded emg')
				print(decoded_emg_dataSet)
				print('decoded flex')
				print(decoded_flex_dataSet)

				emg_dataSet=[]
				flex_dataSet=[]
				start_index=0
				end_index=0
				validation=False

				#check validation of input emg data and get a line of sensor data
				for i in range(len(decoded_emg_dataSet)):
					if decoded_emg_dataSet[i]=='[':
						start_index=i;
						stack=0 
					elif decoded_emg_dataSet[i]==']' and stack==7 :
						end_index=i;
						stack=0
						validation=True

					elif decoded_emg_dataSet[i]==']' and stack!=7 :
						validation=False
						start_index=0
						end_index=0
						stack=0
						print('emg data validation error')
						os.system("pause")
						break

					elif decoded_flex_dataSet[i]==',':
						stack += 1

					else:
						check_data=decoded_flex_dataSet[i]


			else:
				print('no data')


	else:
		print('serial not open')