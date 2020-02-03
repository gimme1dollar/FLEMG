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

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import os


class encoder:
    def __init__(self, queue_list=[], index_dim=1, flex_dim=5, emg_dim=8, seq_length=3, emg_active_dim=8):
        self.queue_list = queue_list
        self.dataSet = []
        self.count = 0

        self.emg_dim = emg_dim
        self.flex_dim = flex_dim
        self.index_dim = index_dim
        self.data_dim = emg_dim + flex_dim
        self.label_dim = flex_dim
        self.seq_length = seq_length
        self.emg_active_dim = emg_active_dim

        self.tmp_I = 0
        self.tmp_E = []
        self.tmp_F = []

emg_dim = 8
flex_dim = 5

EC = encoder(emg_dim=emg_dim, flex_dim=flex_dim)

class plotter:
    def __init__(self, net = "", learning_rate = 0, iteration = 0 ,seq_length = 3 ,stack_dim = 0 ,hidden_dim = 0, rmse = [], prediction = [], label = [], index = [], flex_dim = 5):

        self.net=net
        self.learning_rate=learning_rate
        self.iteration=iteration
        self.seq_length = seq_length
        self.stack_dim=stack_dim
        self.hidden_dim=hidden_dim
        self.rmse=rmse
        self.prediction = prediction
        self.label = label
        self.index = index
        self.flex_dim = flex_dim

    def plot_encoded(self, subplot_row = 2, size = (20,10), figloc = './result/tmp'):
        fig = plt.figure(num=1,figsize=size)
        plt.figure(1)
        #print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")

        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0])+1)//subplot_row ), i+1)

            if i< (len(self.label[0])-self.flex_dim):
                plt.ylim([-20000,20000])
            elif i>=(len(self.label[0])-self.flex_dim):
                plt.ylim([0,500])

            plt.xlabel("time(s)")
            if i< (len(self.label[0])-self.flex_dim):
                plt.plot(self.index[:], self.label[:,i],'r')
            elif i>=(len(self.label[0])-self.flex_dim):
                plt.plot(self.index[:], self.label[:,i],'b')

            if i< (len(self.label[0])-self.flex_dim):
                plt.title(f"emg ch {i+1}")
            elif i>=(len(self.label[0])-self.flex_dim):
                plt.title(f"flex order {i+1-(len(self.label[0])-self.flex_dim)}")
        plt.suptitle(f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()
        plt.pause(2)
        plt.close()

    def plot_filtered(self, subplot_row=2, size=(20, 10), figloc='./result/tmp'):
        fig = plt.figure(num=1, figsize=size)
        plt.figure(1)
        # print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")

        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0]) + 1) // subplot_row), i + 1)

            if i < (len(self.label[0]) - self.flex_dim):
                plt.ylim([-300, 300])
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.ylim([0, 500])

            plt.xlabel("time(s)")
            if i < (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'r')
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'b')

            if i < (len(self.label[0]) - self.flex_dim):
                plt.title(f"emg ch {i + 1}")
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.title(f"flex order {i + 1 - (len(self.label[0]) - self.flex_dim)}")
        plt.suptitle(
            f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()
        plt.pause(2)
        plt.close()

    def plot_scaled(self, subplot_row=2, size=(20, 10), figloc='./result/tmp'):
        fig = plt.figure(num=1, figsize=size)
        plt.figure(1)
        # print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")

        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0]) + 1) // subplot_row), i + 1)

            if i < (len(self.label[0]) - self.flex_dim):
                plt.ylim([0, 1])
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.ylim([0, 1])

            plt.xlabel("time(s)")
            if i < (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'r')
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'b')

            if i < (len(self.label[0]) - self.flex_dim):
                plt.title(f"emg ch {i + 1}")
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.title(f"flex order {i + 1 - (len(self.label[0]) - self.flex_dim)}")
        plt.suptitle(
            f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()
        plt.pause(5)
        plt.close()

    def plot_comparison(self, subplot_row = 2, size = (20,10), figloc = './result/tmp'):
        fig = plt.figure(num=1,figsize=size)
        plt.figure(1)
        #print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")
        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0])+1)//subplot_row ), i+1)
            plt.ylim([0,1])
            plt.xlabel("time(s)")
            plt.plot(self.index[:,:,0], self.prediction[:,i],'--r', self.index[:,:,0], self.label[:,i],'b')
            if i< (len(self.label[0])-self.flex_dim):
                plt.title(f"emg ch {i+1},rmse {self.rmse[i]:0.3f}")
            elif i>=(len(self.label[0])-self.flex_dim):
                plt.title(f"flex order {i+1-(len(self.label[0])-self.flex_dim)},rmse {self.rmse[i]:0.3f}")
        plt.suptitle(f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

    def plot_rmse(self, subplot_row = 2, size = (20,10), figloc = './result/tmp'):
        fig = plt.figure(num=1,figsize=size)
        plt.figure(1)

        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

    def plot_training_graph(self, loss=[], iteration=5000, size=(20,10), figloc = './result'):
        step=list(range(iteration))
        min_loss = np.amin(loss)
        fig = plt.figure(num=2,figsize=size)
        plt.figure(2)
        plt.title(f"Model {self.net},min_loss {min_loss}")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.plot(step,loss)
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

class Sensor_FLEX(threading.Thread):
    def __init__(self, q=queue.Queue(5000), p='COM11', b=115200):
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

FLEX = Sensor_FLEX(q=queue_list[0], p = 'COM11', b = 115200)
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
    p = plotter("Data_Collector", label=prop_data[:, 1:], index=prop_data[:, 0], flex_dim=5)

    # Save Figure
    train_fig_name = now.strftime("%Y%m%d_%H%M_encode_figure")
    train_fig_dir = fig_dir + '/' + train_fig_name
    p.plot_encoded(figloc=train_fig_dir)
    print("Figure " + train_fig_name + " saved\n")







    # EMG 첫번째 채널 이상한 값 없애기
    class cropper:
        def __init__(self, en=encoder(), raw=[]):
            self.raw = raw
            self.encoder = en
            self.data = []
            self.label = []
            self.index = []

        def delete_abnormal_sEMG(self, feature_window=50, data=None):
            if data is not None:
                self.raw = np.asarray(data)

            res = []
            for i in range(len(self.raw)):
                if i > feature_window / 2:
                    mean = np.mean(self.raw[int(i - feature_window / 2): int(i + feature_window / 2), 1])
                    rms = np.sqrt(np.mean(self.raw[int(i - feature_window / 2): int(i + feature_window / 2), 1] ** 2))

                    if mean - rms * 0.8 <= self.raw[i, self.encoder.index_dim] <= mean + rms * 0.8:
                        res.append(self.raw[i])
                    else:
                        pass
            return np.asarray(res)


    CR = cropper(en=EC, raw=prop_data)
    crop_data = CR.delete_abnormal_sEMG()

    print(len(prop_data))
    print(len(crop_data))

    fig = plt.figure(num=1, figsize=(15, 10))
    plt.figure(1)

    plt.subplot(211)
    plt.ylim([-3000, 3000])
    index = prop_data[:, 0]
    sEMGdata = prop_data[:, 1]
    plt.plot(index[:], sEMGdata[:], 'r')

    plt.subplot(212)
    plt.ylim([-3000, 3000])
    index = crop_data[:, 0]
    sEMGdata = crop_data[:, 1]
    plt.plot(index[:], sEMGdata[:], 'r')

    # Save
    csv_name = now.strftime("%Y%m%d_%H%M_crop.csv")
    sav_loc = fig_dir + '/' + csv_name
    f = open(sav_loc, 'w')
    for d in range(len(crop_data)):
        for v in range(len(crop_data[d])):
            f.write(str(crop_data[d][v]))
            if v is not (len(crop_data[d]) - 1):
                f.write(',')
        f.write('\n')
    f.close()
    print("CSV Saved")

    ### Figure
    p = plotter("Data_Collector", label=crop_data[:, 1:], index=crop_data[:, 0], flex_dim=5)

    # Save Figure
    train_fig_name = now.strftime("%Y%m%d_%H%M_crop_figure")
    train_fig_dir = fig_dir + '/' + train_fig_name
    p.plot_encoded(figloc=train_fig_dir)
    print("Figure " + train_fig_name + " saved\n")







    # Filter
    from scipy import signal

    class filtering:
        def __init__(self, en=encoder(), raw=[]):
            self.raw = raw
            self.encoder = en
            self.data = []
            self.label = []
            self.index = []

        def butter_highpass(self, cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def butter_highpass_filter(self, data, cutoff, fs, order=5):
            b, a = self.butter_highpass(cutoff, fs, order=order)
            y = signal.filtfilt(b, a, data)
            return y

        def filter_highpass(self, data=None, cutoff=10, fps=250):
            if data is not None:
                self.raw = np.asarray(data)

            filtered_data = []
            filtered_data.append(self.raw[:, 0])
            for ch in range(emg_dim):
                sEMGdata = self.raw[:, ch + 1]
                sEMGfiltered = self.butter_highpass_filter(sEMGdata, cutoff, fps)
                filtered_data.append(sEMGfiltered)

            for ch in range(flex_dim):
                filtered_data.append(self.raw[:, 9 + ch])

            filtered_data = np.asarray(filtered_data).transpose()
            return filtered_data


    FT = filtering(en=EC, raw=crop_data)
    filtered_data = FT.filter_highpass(cutoff=10)

    # Save
    csv_name = now.strftime("%Y%m%d_%H%M_filter.csv")
    sav_loc = fig_dir + '/' + csv_name
    f = open(sav_loc, 'w')
    for d in range(len(filtered_data)):
        for v in range(len(filtered_data[d])):
            f.write(str(filtered_data[d][v]))
            if v is not (len(filtered_data[d]) - 1):
                f.write(',')
        f.write('\n')
    f.close()
    print("CSV Saved")

    ### Figure
    p = plotter("Data_Collector", label=filtered_data[:, 1:], index=filtered_data[:, 0], flex_dim=5)

    # Save Figure
    train_fig_name = now.strftime("%Y%m%d_%H%M_filter_figure")
    train_fig_dir = fig_dir + '/' + train_fig_name
    p.plot_filtered(figloc=train_fig_dir)
    print("Figure " + train_fig_name + " saved\n")








    # Feature
    class Feature:
        def __init__(self, en=encoder(), raw=[]):
            self.raw = raw
            self.encoder = en
            self.data = []
            self.label = []
            self.index = []

        def scale(self, data=None, emg_max=1500, flex_max=1024, feature_num=5):
            if data is not None:
                self.raw = np.asarray(data)


            # scale EMG raw data
            emg_start_idx = 1
            for i in range(self.encoder.emg_dim):
                self.raw[:, emg_start_idx + i] = self.raw[:, emg_start_idx + i] / emg_max
                self.raw[:, emg_start_idx + i] = self.raw[:, emg_start_idx + i] * 256
                for j in range( len(self.raw[:,1+i]) ):
                    self.raw[j, emg_start_idx+i] = int(self.raw[j, emg_start_idx+i])
                    if self.raw[j, emg_start_idx+i] < -128 :
                        self.raw[j, emg_start_idx + i] = -128
                    elif self.raw[j, emg_start_idx+i] > 128 :
                        self.raw[j, emg_start_idx + i] = 128
                self.raw[:, emg_start_idx + i] = (self.raw[:, emg_start_idx + i] + 128) / 256

            # scale Flex-sensor data
            flex_start_idx = 1 + self.encoder.emg_dim
            for i in range(self.encoder.flex_dim):
                if i == 0:
                    self.raw[:, flex_start_idx + i] = np.absolute(self.raw[:, flex_start_idx + i])
                    self.raw[:, flex_start_idx + i] /= flex_max
                    self.raw[:, flex_start_idx + i] = (self.raw[:, flex_start_idx + i] - 0.15) / (0.35 - 0.15)
                else:
                    self.raw[:, flex_start_idx+i] = np.absolute(self.raw[:, flex_start_idx + i])
                    self.raw[:, flex_start_idx+i] /= flex_max
                    self.raw[:, flex_start_idx+i] = (self.raw[:, flex_start_idx + i] - 0.15) / (0.4 - 0.15)


    SC = Feature(en=EC)
    SC.scale(filtered_data)
    scale_data = SC.raw
    print(f"scaled:{scale_data[0]}")

    # Save
    csv_name = now.strftime("%Y%m%d_%H%M_scale.csv")
    sav_loc = fig_dir + '/' + csv_name
    f = open(sav_loc, 'w')
    for d in range(len(scale_data)):
        for v in range(len(scale_data[d])):
            f.write(str(scale_data[d][v]))
            if v is not (len(scale_data[d]) - 1):
                f.write(',')
        f.write('\n')
    f.close()
    print("CSV Saved")

    ### Figure
    p = plotter("Data_Collector", label=scale_data[:, 1:], index=scale_data[:, 0], flex_dim=5)

    # Save Figure
    train_fig_name = now.strftime("%Y%m%d_%H%M_scale_figure")
    train_fig_dir = fig_dir + '/' + train_fig_name
    p.plot_scaled(figloc=train_fig_dir)
    print("Figure " + train_fig_name + " saved\n")
