import numpy as np
import time
from datetime import datetime

print("\n\nStart")

# Open
file_time = '202001041700'
location = '40msCollector' + '_' + file_time + '.txt'
file=open(location,'r')
raw_data=file.read()
file.close()

dataset = []
raw_data = raw_data.split('INDEX')
for d in range( len(raw_data) - 1 ):
  d += 1
  tmp = []

  index = raw_data[d].split('_FLEX_')[0]
  data = raw_data[d].split('_FLEX_')[1]
  flex = data.split('_EMG_')[0]
  emg = data.split('_EMG_')[1]

  # Make Index
  sec = float(index.split('\'')[1]) # second part
  #print(f"{d} Index second : {sec}")
  for t in range(10):
    tmp.append([sec + t*0.004])

  # EMG
  emg_set = emg.split(']')
  #print(f"{d} EMG samples : {len(emg_set)}")
  
  emg_parts = []
  for e in range(len(emg_set)) :
    if len(emg_set[e]) > 80 and len(emg_set[e]) < 120: 
      emg_parts.append(emg_set[e].split(','))
  
  for t in range( len(emg_parts) ):
    # 맨 처음 [ 떼어주기
    emg_parts[t][0] = emg_parts[t][0][1:]

    # b' 포함돼있는 것들 때문에 채널 하나 날라가는 문제 해결
    flag_b = False
    for v in range( len(emg_parts[t]) ) :
      if emg_parts[t][v].find('b\'') != -1:
        emg_parts[t][v] = emg_parts[t][v][3:]

        flag_b = True
        break
      
    if flag_b is True and v != 0 :
      emg_parts[t][v-1] = emg_parts[t][v-1][: len(emg_parts[t][v-1])-1 ] + emg_parts[t][v]

      for v_ in range( len(emg_parts[t]) - v - 1 ) :
        emg_parts[t][v_ + v] = emg_parts[t][v_ + v + 1]

    # 데이터셋에 들어갈 tmp에 emg데이터 집어넣기    
    if t < 10:
      for v in range( 8 ):
       try:
         value = float(emg_parts[t][v])
         tmp[t].append(value)
       except:
         #print(f"# {d} {t} has ValueError {emg_parts[t][v]}")
         pass

      #print(f"{tmp[t][0]} EMG samples : {v}")

  # FLEX 
  flex_set = flex.split('\\n')
  #print(f"{d} FLEX samples : {len(flex_set)}")
  
  flex_parts = []
  for f in range(len(flex_set)) :
    if len(flex_set[f]) == 19:
      flex_parts.append(flex_set[f].split(','))

  for t in range( len(flex_parts) ):
    if t < 10:
      for v in range( len(flex_parts[t]) ):
        value = float(flex_parts[t][v])
        tmp[t].append(value)

      #print(f"{tmp[t][0]} FLEX samples : {v}")

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

