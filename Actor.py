import serial
import os

"""
From Inferred FLEX value, control MOTOR
PID algorithm exploited
"""

class Actor:
	def __init__(self, flex_size, motor_size, target_state = [], current_state = []):
		self.target_state = target_state 
		self.curret_state = current_state

		self.flex_channels = flex_size
		self.flex_port = None
		self.flex_port_open = false

		self.motor_channels = motor_size
		self.motor_port = None
		self.motor_port_open = false


	def open_port(self, flex_port_name, motor_port_name):
		self.flex_port = serial.Serial(flex_port_name, 115200, timeout=2)
		self.motor_port = serial.Serial(motor_port_name, 19200, timeout=2)
	
		if self.flex_port.is_open:
			self_flex_port_open = True
		else:
			print('FLEX serial not open')
		if self.motor_port.is_open:
			self_motor_port_open = True
		else:
			print('MOTOR serial not open')

	def close_port(self):
		if self.flex_port.is_open:
			self.flex_port.close()
			self.flex_port_open = False

		if self.motor_port.is_open:
			self.motor_port.close()
			self.motor_port_open = False
	
	def update_target_state(self, target_value = []):
		self.target_state= target_state

	def update_current_state(self, current_state = None):
		if current_state is not None:
			current_state = current_state
		else:
			# get current_state of FLEX from serial port

	def control_motors(self):
		#control motor to achieve the target
		