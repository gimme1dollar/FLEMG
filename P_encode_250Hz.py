import numpy as np
import time
from datetime import datetime

print("\n\nStart")

# Open
file_time = '202001041534'
location = '250Hz_' + file_time + '.txt'
file=open(location,'r')
raw_data=file.read()
file.close()

dataset = []
raw_data = raw_data.split('INDEX')
for d in range( len(raw_data) - 1 ):
  d+=1
  tmp = []

  index = raw_data[d].split('_FLEX_')[0]
  data = raw_data[d].split('_FLEX_')[1]
  flex = data.split('_EMG_')[0]
  emg = data.split('_EMG_')[1]

  # Make Index
  sec = float(index.split('\'')[1]) # second part
  for t in range(100):
    tmp.append([sec + t*0.004])

  # EMG
  emg_set = emg.split(']')
  #print(f"{d} EMG samples : {len(emg_set)}")
  
  emg_parts = []
  for e in range(len(emg_set)) :
    if len(emg_set[e]) > 80 and len(emg_set[e]) < 100: 
      emg_parts.append(emg_set[e].split(','))

  for t in range( len(emg_parts) ):
    emg_parts[t][0] = emg_parts[t][0][1:]
    for v in range( len(emg_parts[t]) ):
      try:
        value = float(emg_parts[t][v])
        tmp[t].append(value)
      except:
        #print(f"# {d} {t} has ValueError {emg_parts[t][v]}")
        pass
      
  # FLEX 
  flex_set = flex.split('\\n')
  #print(f"{d} FLEX samples : {len(flex_set)}")
  
  flex_parts = []
  for f in range(len(flex_set)) :
    if len(flex_set[f]) == 19:
      flex_parts.append(flex_set[f].split(','))

  for t in range( len(flex_parts) ):
    if t < 100:
      for v in range( len(flex_parts[t]) ):
        value = float(flex_parts[t][v])
        tmp[t].append(value)

  # FLEX + EMG into DataSet
  for s in range( len(tmp) ):
    dataset.append( tmp[s] )


# Save
sav_loc = 'encode_' + file_time + '.csv'
f = open(sav_loc, 'w')
for d in range( len(dataset) ) :
  for v in range( len(dataset[d]) ) :
    f.write(str(dataset[d][v]))
    if v is not ( len(dataset[d])-1 ):
      f.write(',')
  f.write('\n')
f.close()

print("Saved")
