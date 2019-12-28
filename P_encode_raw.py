import numpy as np
import time
from datetime import datetime

print("Start")

start = time.time()
now = datetime.now()
maxsize = 0xffffffff

location = '250Hz_' + '201912262233.txt'
deli = '_delimeter_'
raw_data = np.loadtxt(location, delimiter = deli, dtype=str)

dataset = []
for d in range(len(raw_data)):
  tmp = []

  index = raw_data[d][0]
  sec = int(index.split('.')[0][2]) # second part
  for t in range(250):
    tmp.append([sec + t*0.004])

  file=open('emg0.txt','r')
  emg=file.read()
  file.close()
  emg_parts = str(emg).split('\\xa0')
  for e in range( len(emg_parts) ):
    if (emg_parts[e].find('\\xc0')) is not -1:
      emg_values = emg_parts[e].split('\\x')
      print(emg_parts[e])
      print(emg_values)

      scale = 16 ## equals to hexadecimal 
      num_of_bits = 8 
      for v in range( len(emg_values)-1 ):
        print(emg_values[v+1])

        import binascii
        print( ''.join( byte_to_binary(ord(b)) for b in binascii.unhexlify(emg_values[v+1]) ) )

  if d == 0:
    print(emg_parts[e])
    print(emg_values)
  
  flex = raw_data[d][1]
  flex_set = flex.split('\\n')
  flex_parts = []
  for f in range(len(flex_set)) :
    if len(flex_set[f]) == 19:
      flex_parts.append(flex_set[f].split(','))
  for t in range( len(flex_parts) ): 
    if t < 250: # pick 250 data
      tmp[t] += flex_parts[t]

  dataset+=tmp
