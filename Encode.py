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
            plt.subplot(subplot_row, int((len(self.label[0])+1)/subplot_row ), i+1)

            if i>=(len(self.label[0])-self.flex_dim):
                plt.ylim([0,500])

            plt.xlabel("time(s)")
            if i< (len(self.label[0])-self.flex_dim):
                plt.plot(self.index[:], self.label[:,i],'r',linewidth=0.5)
            elif i>=(len(self.label[0])-self.flex_dim):
                plt.plot(self.index[:], self.label[:,i],'b',linewidth=0.5)

            if i< (len(self.label[0])-self.flex_dim):
                plt.title(f"emg ch {i+1}")
            elif i>=(len(self.label[0])-self.flex_dim):
                plt.title(f"flex order {i+1-(len(self.label[0])-self.flex_dim)}")
        plt.suptitle(f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        plt.ion()
        plt.show()
        plt.pause(1)
        fig.savefig(figloc, dpi=fig.dpi)


    def plot_filtered(self, subplot_row=2, size=(20, 10), figloc='./result/tmp'):
        fig = plt.figure(num=2, figsize=size)
        plt.figure(2)
        # print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")

        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0]) + 1) / subplot_row), i + 1)

            if i < (len(self.label[0]) - self.flex_dim):
                plt.ylim([-1500, 1500])
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.ylim([0, 500])

            plt.xlabel("time(s)")
            if i < (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'r',linewidth=0.5)
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'b',linewidth=0.5)

            if i < (len(self.label[0]) - self.flex_dim):
                plt.title(f"emg ch {i + 1}")
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.title(f"flex order {i + 1 - (len(self.label[0]) - self.flex_dim)}")
        plt.suptitle(
            f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        plt.ion()
        plt.show()
        plt.pause(1)
        fig.savefig(figloc, dpi=fig.dpi)


    def plot_scaled(self, subplot_row=2, size=(20, 10), figloc='./result/tmp'):
        fig = plt.figure(num=3, figsize=size)
        plt.figure(3)
        # print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")

        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0]) + 1) / subplot_row), i + 1)

            if i < (len(self.label[0]) - self.flex_dim):
                plt.ylim([0, 1])
                pass
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.ylim([0, 1])
                pass

            plt.xlabel("time(s)")
            if i < (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'r',linewidth=0.5)
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'b',linewidth=0.5)

            if i < (len(self.label[0]) - self.flex_dim):
                plt.title(f"emg ch {i + 1}")
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.title(f"flex order {i + 1 - (len(self.label[0]) - self.flex_dim)}")
        plt.suptitle(
            f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        plt.ion()
        plt.show()
        plt.pause(1)
        fig.savefig(figloc, dpi=fig.dpi)

    def plot_featured(self, subplot_row=2, size=(20, 10), figloc='./result/tmp'):
        fig = plt.figure(num=4, figsize=size)
        plt.figure(4)
        # print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")

        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0]) + 1) / subplot_row), i + 1)

            if i < (len(self.label[0]) - self.flex_dim):
                #plt.ylim([0, 1])
                pass
            elif i >= (len(self.label[0]) - self.flex_dim):
                #plt.ylim([0, 1])
                pass

            plt.xlabel("time(s)")
            if i < (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'r',linewidth=0.5)
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.plot(self.index[:], self.label[:, i], 'b',linewidth=0.5)

            if i < (len(self.label[0]) - self.flex_dim):
                plt.title(f"emg ch {i + 1}")
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.title(f"flex order {i + 1 - (len(self.label[0]) - self.flex_dim)}")
        plt.suptitle(
            f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        plt.ion()
        plt.show()
        plt.pause(1)
        fig.savefig(figloc, dpi=fig.dpi)


emg_dim = 8
flex_dim = 5

subject_name = "YHyu"

EC = encoder(emg_dim=emg_dim, flex_dim=flex_dim)


# Instantiation
start = time.time()
now = datetime.now()

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


if True:
    ### Save Raw
    time_name = "20200129_2127"

    #training_name = "20200129_2127_encode"
    #testing_name = "20200129_2201_encode"
    txt_name = time_name + "_raw.txt"
    sav_loc = fig_dir + '/' + txt_name
    #np.savetxt(sav_loc, np.asarray(dataSet), fmt='%s', delimiter='___')
    #print("TXT saved")


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
    csv_name = time_name + "_encode.csv"
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
    train_fig_name = time_name + "_encode_figure"
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

                    if mean - rms * 0.7 <= self.raw[i, self.encoder.index_dim] <= mean + rms * 0.7:
                        res.append(self.raw[i])
                    else:
                        pass
            return np.asarray(res)


    CR = cropper(en=EC, raw=prop_data)
    crop_data = CR.delete_abnormal_sEMG()

    print(len(prop_data))
    print(len(crop_data))


    # Save
    csv_name = time_name + "_crop.csv"
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
    train_fig_name = time_name + "_crop_figure"
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

        def butter_highpass(self, cutoff, fs, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def butter_highpass_filter(self, data, cutoff, fs, order=4):
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
    csv_name = time_name + "_filter.csv"
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
    print("filtered label : " + str(len(filtered_data[:, 1:][0])))
    p = plotter("Data_Collector", label=filtered_data[:, 1:], index=filtered_data[:, 0], flex_dim=5)

    # Save Figure
    train_fig_name = time_name + "_filter_figure"
    train_fig_dir = fig_dir + '/' + train_fig_name
    p.plot_filtered(figloc=train_fig_dir)
    print("Figure " + train_fig_name + " saved\n")


    class feature_extractor:
        def __init__(self, en = encoder(), raw = []):
            self.raw = raw
            self.encoder = en
            self.data = []
            self.label = []
            self.index = []

        def feature_average(self, average_unit1 = 25, average_unit2 = 250, data = None):
            if data is not None:
                self.raw = np.asarray(data)

            res = []
            for i in range(len(self.raw)):
                if i > average_unit2:
                    tmp = [self.raw[i, 0]]
                    for ch in range(self.encoder.emg_dim):
                        tmp.append( self.raw[i, self.encoder.index_dim+ch] )

                    for ch in range(self.encoder.emg_dim):
                        avg1 = 0
                        for r in range(average_unit1):
                            avg1 += self.raw[i-r, self.encoder.index_dim+ch]
                        tmp.append( avg1/average_unit1 )

                    for ch in range(self.encoder.emg_dim):
                        avg2 = 0
                        for r in range(average_unit2):
                            avg2 += self.raw[i-r, self.encoder.index_dim+ch]
                        tmp.append( avg2/average_unit2 )

                    tmp = np.concatenate( (tmp, self.raw[i, 1+self.encoder.emg_dim:]), axis=None)

                    res.append(tmp.tolist())

                if i % 100 == 0:
                    print(f"{i}/{len(self.raw)} Feature Extracting...")

            return np.asarray(res)


    FE = feature_extractor(en=EC, raw=filtered_data)
    #FE = feature_extractor(en=Encoder, raw=raw_data)
    feat_data = FE.feature_average()
    print(f"feat:{feat_data[0]}")
    print(len(feat_data[0]))


    # Save
    csv_name = time_name + "_feature.csv"
    sav_loc = fig_dir + '/' + csv_name
    f = open(sav_loc, 'w')
    for d in range(len(feat_data)):
        for v in range(len(feat_data[d])):
            f.write(str(feat_data[d][v]))
            if v is not (len(feat_data[d]) - 1):
                f.write(',')
        f.write('\n')
    f.close()
    print("CSV Saved")

    print("feature label : "+str(len(feat_data[:, 1:][0])))
    p = plotter("Data_Collector", label=feat_data[:, 1:], index=feat_data[:, 0], flex_dim=5)

    # Save Figure
    train_fig_name = time_name + "_feature_figure"
    train_fig_dir = fig_dir + '/' + train_fig_name
    p.plot_featured(figloc=train_fig_dir)
    print("Figure " + train_fig_name + " saved\n")


    # Scaler
    class Scaler:
        def __init__(self, en=encoder(), featured=[]):
            self.featured = featured
            self.encoder = en
            self.data = []
            self.label = []
            self.index = []

        def scale(self, data=None, emg_max=1500, flex_max=1024, feature_num=5):
            if data is not None:
                self.featured = np.asarray(data)

            # scale EMG featured data
            emg_start_idx = 1

            for i in range(self.encoder.emg_dim):
                self.featured[:, emg_start_idx + i] = self.featured[:, emg_start_idx + i] / emg_max
                self.featured[:, emg_start_idx + i] = self.featured[:, emg_start_idx + i] * 256
                for j in range(len(self.featured[:, 1 + i])):
                    self.featured[j, emg_start_idx + i] = int(self.featured[j, emg_start_idx + i])
                    if self.featured[j, emg_start_idx + i] < -128:
                        self.featured[j, emg_start_idx + i] = -128
                    elif self.featured[j, emg_start_idx + i] > 128:
                        self.featured[j, emg_start_idx + i] = 128
                # self.featured[:, emg_start_idx + i] = (self.featured[:, emg_start_idx + i] + 128) / 256
                self.featured[:, emg_start_idx + i] = np.absolute(self.featured[:, emg_start_idx + i] / 256)

            for i in range(self.encoder.emg_dim,self.encoder.emg_dim*2):
                self.featured[:, emg_start_idx + i] = self.featured[:, emg_start_idx + i]*20 / (emg_max)
                #print("First : "+str(self.featured[:, emg_start_idx + i]))
                self.featured[:, emg_start_idx + i] = self.featured[:, emg_start_idx + i] * 256
                for j in range(len(self.featured[:, 1 + i])):
                    self.featured[j, emg_start_idx + i] = int(self.featured[j, emg_start_idx + i])
                    if self.featured[j, emg_start_idx + i] < -128:
                        self.featured[j, emg_start_idx + i] = -128
                    elif self.featured[j, emg_start_idx + i] > 128:
                        self.featured[j, emg_start_idx + i] = 128
                # self.featured[:, emg_start_idx + i] = (self.featured[:, emg_start_idx + i] + 128) / 256
                self.featured[:, emg_start_idx + i] = np.absolute(self.featured[:, emg_start_idx + i] / 256)

            for i in range(self.encoder.emg_dim*2,self.encoder.emg_dim*3):
                self.featured[:, emg_start_idx + i] = self.featured[:, emg_start_idx + i]*250 / (emg_max)
                self.featured[:, emg_start_idx + i] = self.featured[:, emg_start_idx + i] * 256
                for j in range(len(self.featured[:, 1 + i])):
                    self.featured[j, emg_start_idx + i] = int(self.featured[j, emg_start_idx + i])
                    if self.featured[j, emg_start_idx + i] < -128:
                        self.featured[j, emg_start_idx + i] = -128
                    elif self.featured[j, emg_start_idx + i] > 128:
                        self.featured[j, emg_start_idx + i] = 128
                # self.featured[:, emg_start_idx + i] = (self.featured[:, emg_start_idx + i] + 128) / 256
                self.featured[:, emg_start_idx + i] = np.absolute(self.featured[:, emg_start_idx + i] / 256)

            # scale Flex-sensor data
            flex_start_idx = 1 + self.encoder.emg_dim*3
            '''
            for i in range(self.encoder.flex_dim):
                if i == 0:
                    self.featured[:, flex_start_idx + i] = np.absolute(self.featured[:, flex_start_idx + i])
                    self.featured[:, flex_start_idx + i] /= flex_max
                    self.featured[:, flex_start_idx + i] = (self.featured[:, flex_start_idx + i] - 0.15) / (0.35 - 0.15)
                else:
                    self.featured[:, flex_start_idx + i] = np.absolute(self.featured[:, flex_start_idx + i])
                    self.featured[:, flex_start_idx + i] /= flex_max
                    self.featured[:, flex_start_idx + i] = (self.featured[:, flex_start_idx + i] - 0.15) / (0.4 - 0.15)
            '''

    SC = Scaler(en=EC)
    SC.scale(feat_data)
    scale_data = SC.featured
    print(f"scaled:{scale_data[0]}")

    # Save
    csv_name = time_name + "_scale.csv"
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
    print("scale label : " + str(len(scale_data[:, 1:][0])))
    p = plotter("Data_Collector", label=scale_data[:, 1:], index=scale_data[:, 0], flex_dim=5)

    # Save Figure
    train_fig_name = time_name + "_scale_figure"
    train_fig_dir = fig_dir + '/' + train_fig_name
    p.plot_scaled(figloc=train_fig_dir)
    print("Figure " + train_fig_name + " saved\n")