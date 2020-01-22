try:
  import queue
except ImportError:
  import Queue as queue
import numpy as np
import time
from datetime import datetime
import serial
import threading
try:
  import queue
except ImportError:
  import Queue as queue
import time
import os, sys
from _modules import Analysis

class Sensor_FLEX(threading.Thread):
    def __init__(self, q=queue.Queue(5000), p='COM5', b=115200):
        threading.Thread.__init__(self)
        self._exit = False

        self.storage = q
        self.port = serial.Serial(p, b, timeout=1) # (port name, baudrate, timeout)
        

    def run(self):
        while self.port.is_open:
            data = self.port.read(1000)
            self.storage.put(data)


    def exit(self):
        self._exit = True

class Sensor_EMG(threading.Thread):
    def __init__(self, q=queue.Queue(5000), p='COM5', b=115200):
        threading.Thread.__init__(self)
        self._exit = False
        self.storage = q
        self.port = serial.Serial(p, b, timeout=1)  # (port name, baudrate, timeout)

    def send(self, char):
        if self.port.is_open:
            self.port.write(bytearray(char, 'ascii'))
            print(f"Serial Write {char}")
        else:
            print("Serial closed")

    def run(self):
        while self.port.is_open:
            data = self.port.read(5000)
            self.storage.put(data)

    def exit(self):
        self._exit = True


# Instantiation
start = time.time()
now = datetime.now()
maxsize = 0xffffffff
queue_list = [queue.Queue(maxsize), queue.Queue(maxsize)]
subject_name = "JYLee"

FLEX = Sensor_FLEX(q=queue_list[0], p = 'COM7', b = 115200)
EMG = Sensor_EMG(q=queue_list[1], p = 'COM8', b = 115200)

FLEX.start()
EMG.start()
start = time.time()

fig_dir = "./data/" + subject_name
try:
    # Create target Directory
    os.mkdir(fig_dir)
    print("Directory ", fig_dir, " Created\n")
    fig_dir = "./data/" + subject_name
    try:
        os.mkdir(fig_dir)
        print("Directory", fig_dir, "Created\n")
    except FileExistsError:
        print("Directory", fig_dir, "already exists\n")

except FileExistsError:
    fig_dir = "./data/" + subject_name
    try:
        os.mkdir(fig_dir)
        print("Directory", fig_dir, "Created\n")
    except FileExistsError:
        print("Directory", fig_dir, "already exists\n")

dataSet = []
try:
    while True:
        tmp = []
        queue_list[0].queue.clear()
        queue_list[1].queue.clear()
        time.sleep(0.4)

        index = time.time() - start
        print("INDEX_"+str(index))
        tmp.append("INDEX_b\'" + str(index)+"\'")

        f_q = list(queue_list[0].queue)
        #print(f"f_q {f_q}")
        tmp.append("FLEX_" + str(f_q) )

        e_q = list(queue_list[1].queue)
        #print(f"e_q {e_q}")
        tmp.append( "EMG_" + str(e_q) )

        #print(f"tmp {tmp}")
        dataSet.append(tmp)

        #clear

except KeyboardInterrupt:
    print("keyboard interuupt")

    EMG.exit()
    FLEX.exit()

    ### Save Raw
    txt_name = now.strftime("%Y%m%d_%H%M_raw.txt")
    sav_loc = fig_dir + '/' + txt_name
    np.savetxt(sav_loc, np.asarray(dataSet), fmt='%s', delimiter='___')
    print("TXT saved")


    ### Encode
    file = open(sav_loc, 'r')
    raw_data = file.read()
    file.close()

    dataset = []
    raw_data = raw_data.split('INDEX')
    for d in range(len(raw_data) - 1):
        d += 1
        tmp = []

        index = raw_data[d].split('_FLEX_')[0]
        data = raw_data[d].split('_FLEX_')[1]
        flex = data.split('_EMG_')[0]
        emg = data.split('_EMG_')[1]

        # Make Index
        sec = float(index.split('\'')[1])  # second part
        print(f"{d} Index second : {sec}")
        for t in range(100):
            tmp.append([sec + t * 0.004])

        # EMG
        emg_set = emg.split(']')
        print(f"{d} EMG samples : {len(emg_set)}")

        emg_parts = []
        for e in range(len(emg_set)):
            emg_parts.append(emg_set[e].split(','))

        for t in range(len(emg_parts)):
            # 맨 처음 [ 떼어주기
            emg_parts[t][0] = emg_parts[t][0][1:]

            # b' 포함돼있는 것들 때문에 채널 하나 날라가는 문제 해결
            flag_b = False
            for v in range(len(emg_parts[t])):
                if emg_parts[t][v].find('b\'') != -1:
                    emg_parts[t][v] = emg_parts[t][v][3:]

                    flag_b = True
                    break

            if flag_b is True and v != 0:
                emg_parts[t][v - 1] = emg_parts[t][v - 1][: len(emg_parts[t][v - 1]) - 1] + emg_parts[t][v]

                for v_ in range(len(emg_parts[t]) - v - 1):
                    emg_parts[t][v_ + v] = emg_parts[t][v_ + v + 1]

            # 데이터셋에 들어갈 tmp에 emg데이터 집어넣기
            if t < 100:
                for v in range(8):
                    try:
                        value = float(emg_parts[t][v])
                        tmp[t].append(value)
                    except:
                        # print(f"# {d} {t} has ValueError {emg_parts[t][v]}")
                        pass

                # print(f"{tmp[t][0]} EMG samples : {v}")

        # FLEX
        flex_set = flex.split('\\n')
        print(f"{d} FLEX samples : {len(flex_set)}")

        flex_parts = []
        for f in range(len(flex_set)):
            if len(flex_set[f]) == 19:
                flex_parts.append(flex_set[f].split(','))

        for t in range(len(flex_parts)):
            if t < 100:
                for v in range(len(flex_parts[t])):
                    try:
                        value = float(flex_parts[t][v])
                        tmp[t].append(value)
                    except:
                        pass
                # print(f"{tmp[t][0]} FLEX samples : {v}")

        # FLEX + EMG into DataSet
        for s in range(len(tmp)):
            dataset.append(tmp[s])

    # Trimm Wrong Data
    full_idx = []
    for d in range(len(dataset)):
        if len(dataset[d]) == 14:
            full_idx.append(d)

    prop_data = np.asarray(list(map(dataset.__getitem__, full_idx)))

    for d in range(len(prop_data)):
        for v in range(len(prop_data[d])):
            try:
                prop_data[d][v] = float(prop_data[d][v])
            except:
                print(f"{d} {v} : {prop_data[d][v]}")

    # Save
    csv_name = now.strftime("%Y%m%d_%H%M_encode.csv")
    sav_loc = fig_dir + '/' + csv_name
    f = open(sav_loc, 'w')
    for d in range(len(prop_data)):
        for v in range(len(prop_data[d])):
            f.write(str(prop_data[d][v]))
            if v is not (len(prop_data[d]) - 1):
                f.write(',')
        f.write('\n')
    f.close()
    print("CSV Saved")

    ### Figure
    p = Analysis.plotter("Data_Collector", label=prop_data[:,1:], index=prop_data[:,0], flex_dim=5)

    # Save Figure
    train_fig_name = now.strftime("%Y%m%d_%H%M_figure")
    train_fig_dir = fig_dir + '/' + train_fig_name
    p.plot_encoded(figloc=train_fig_dir)
    print("Figure " + train_fig_name + " saved\n")
