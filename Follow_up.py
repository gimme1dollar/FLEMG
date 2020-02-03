import os
from os import path
# from google.colab import drive
from datetime import datetime
import numpy as np
# import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import math

'''
try:
  %tensorflow_version 1.x
except:
  pass
'''
import tensorflow as tf

now = datetime.now()

notebook_path_name = os.path.dirname(__file__)
# notebook_path_name = '각자 알아서 맞출것'

if not path.exists(notebook_path_name):
    print('Check your google drive directory. See you file explorer')
with open(path.join(notebook_path_name, "test.txt"), "w") as f:
    f.write("test well saved!!!")

subject_name = "JYLee"
network_name = "Network_default_feature_average"
time_name = now.strftime("%Y%m%d_%H%M")
model_name = notebook_path_name + '/model/' + network_name

try:
    # Create target Directory
    os.makedirs(model_name)
    print("Directory ", model_name, " Created\n")
    subject_dir = model_name + '/' + subject_name
    try:
        os.makedirs(subject_dir)
        print("Directory", subject_dir, "Created\n")
    except FileExistsError:
        print("Directory ", subject_dir, " already exists\n")
except FileExistsError:
    subject_dir = model_name + '/' + subject_name
    try:
        os.makedirs(subject_dir)
        print("Directory", subject_dir, "Created\n")
    except FileExistsError:
        print("Directory ", subject_dir, " already exists\n")

model_dir = subject_dir + '/' + time_name

fig_name = notebook_path_name + "/result/" + network_name
try:
    # Create target Directory
    os.makedirs(fig_name)
    print("Directory ", fig_name, " Created\n")
    subject_dir = fig_name + "/" + subject_name
    try:
        os.makedirs(subject_dir)
        print("Directory", subject_dir, "Created\n")
    except FileExistsError:
        print("Directory", subject_dir, "already exists\n")

except FileExistsError:
    subject_dir = fig_name + "/" + subject_name
    try:
        os.makedirs(subject_dir)
        print("Directory", subject_dir, "Created\n")
    except FileExistsError:
        print("Directory", subject_dir, "already exists\n")
fig_dir = subject_dir








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


subject_name = "JYLee"
subject_dir = notebook_path_name + '/data/' + subject_name
target_dir = subject_dir + '/2_index'
training_name = "20200203_1821_encode"
testing_name = "20200129_2146_encode"
training_file = target_dir + "/" + training_name + ".csv"
testing_file = target_dir + "/" + testing_name + ".csv"


class data_loader:
    def __init__(self, en=encoder(), raw=[]):
        self.raw = raw
        self.encoder = en
        self.data = []
        self.label = []
        self.index = []

    def load(self, location='default', delimiter=','):
        self.raw = np.loadtxt(location, delimiter=delimiter)


emg_dim = 8
flex_dim = 5

EC = encoder(emg_dim=emg_dim, flex_dim=flex_dim)
DL = data_loader(en=EC)
DL.load(training_file)
raw_data = DL.raw

print(f"{np.asarray(raw_data).shape}")
print(f"raw: {raw_data[0]}")


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


CR = cropper(en=EC, raw=raw_data)
crop_data = CR.delete_abnormal_sEMG()

print(len(raw_data))
print(len(crop_data))

fig = plt.figure(num=1, figsize=(15, 10))
plt.figure(1)

plt.subplot(211)
plt.ylim([-3000, 3000])
index = raw_data[:, 0]
sEMGdata = raw_data[:, 1]
plt.plot(index[:], sEMGdata[:], 'r')

plt.subplot(212)
plt.ylim([-3000, 3000])
index = crop_data[:, 0]
sEMGdata = crop_data[:, 1]
plt.plot(index[:], sEMGdata[:], 'r')


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

index = crop_data[:, 0]
sEMGdata = crop_data[:, 2]
flexdata = crop_data[:, 12]

fig = plt.figure(num=2, figsize=(15, 10))
plt.figure(2)
plt.subplot(311)
plt.ylim([2500, 5000])
plt.plot(index[:], sEMGdata[:], 'r')

plt.subplot(312)
plt.ylim([-500, 500])
plt.plot(index[:], filtered_data[:, 2], 'r')

plt.subplot(313)
plt.ylim([200, 500])
plt.plot(index[:], flexdata[:], 'b')


class scaler:
    def __init__(self, en=encoder(), raw=[]):
        self.raw = raw
        self.encoder = en
        self.data = []
        self.label = []
        self.index = []

    def scale(self, data=None, emg_max=187500.016, flex_max=1024, feature_num=5):
        if data is not None:
            self.raw = np.asarray(data)
        emg_max = 500
        # scale EMG raw data
        self.raw[:, self.encoder.index_dim:self.encoder.index_dim + self.encoder.emg_dim] = np.absolute(
            self.raw[:, self.encoder.index_dim:self.encoder.index_dim + self.encoder.emg_dim])
        self.raw[:, self.encoder.index_dim:self.encoder.index_dim + self.encoder.emg_dim] = self.raw[:,
                                                                                            self.encoder.index_dim:self.encoder.index_dim + self.encoder.emg_dim] / emg_max
        self.raw[:,
        self.encoder.index_dim + self.encoder.emg_active_dim:self.encoder.index_dim + self.encoder.emg_dim] = 0
        # self.raw[:,self.encoder.index_dim:self.encoder.index_dim+self.encoder.emg_dim]=(self.raw[:,self.encoder.index_dim:self.encoder.index_dim+self.encoder.emg_dim]--0.08)/(0.08--0.08)

        # scale Flex-sensor data
        flex_start_idx = 1 + self.encoder.emg_dim
        self.raw[:, flex_start_idx:] = np.absolute(self.raw[:, flex_start_idx:])
        self.raw[:, flex_start_idx:] /= flex_max
        self.raw[:, flex_start_idx:] = (self.raw[:, flex_start_idx:] - 0.15) / (0.4 - 0.15)


SC = scaler(en=EC)
SC.scale(filtered_data)
scale_data = SC.raw
print(f"scaled:{scale_data[0]}")

index = scale_data[:, 0]

fig = plt.figure(num=3, figsize=(15, 10))
plt.figure(3)

plt.subplot(13, 1, 1)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 1], 'b')

plt.subplot(13, 1, 2)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 2], 'b')

plt.subplot(13, 1, 3)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 3], 'b')

plt.subplot(13, 1, 4)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 4], 'b')

plt.subplot(13, 1, 5)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 5], 'b')

plt.subplot(13, 1, 6)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 6], 'b')

plt.subplot(13, 1, 7)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 7], 'b')

plt.subplot(13, 1, 8)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 8], 'b')

plt.subplot(13, 1, 9)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 9], 'b')

plt.subplot(13, 1, 10)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 10], 'b')

plt.subplot(13, 1, 11)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 11], 'b')

plt.subplot(13, 1, 12)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 12], 'b')

plt.subplot(13, 1, 13)
plt.ylim([0, 1.2])
plt.plot(index[:], scale_data[:, 13], 'b')


class feature_extractor:
    def __init__(self, en=encoder(), raw=[]):
        self.raw = raw
        self.encoder = en
        self.data = []
        self.label = []
        self.index = []

    def feature_average(self, average_unit1=25, average_unit2=250, data=None):
        if data is not None:
            self.raw = np.asarray(data)

        res = []
        for i in range(len(self.raw)):
            if i > average_unit2:
                tmp = [self.raw[i, 0]]
                for ch in range(self.encoder.emg_dim):
                    tmp.append(self.raw[i, self.encoder.index_dim + ch])

                for ch in range(self.encoder.emg_dim):
                    avg1 = 0
                    for r in range(average_unit1):
                        avg1 += self.raw[i - r, self.encoder.index_dim + ch]
                    tmp.append(avg1 / average_unit1)

                for ch in range(self.encoder.emg_dim):
                    avg2 = 0
                    for r in range(average_unit2):
                        avg2 += self.raw[i - r, self.encoder.index_dim + ch]
                    tmp.append(avg2 / average_unit2)

                tmp = np.concatenate((tmp, self.raw[i, 1 + self.encoder.emg_dim:]), axis=None)

                res.append(tmp.tolist())

        return np.asarray(res)


FE = feature_extractor(en=EC, raw=scale_data)
# FE = feature_extractor(en=Encoder, raw=raw_data)
feat_data = FE.feature_average()
print(f"feat:{feat_data[0]}")

index = feat_data[:, 0]
fig = plt.figure(num=4, figsize=(15, 10))
plt.figure(4)

plt.subplot(13, 1, 1)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 17], 'b')

plt.subplot(13, 1, 2)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 18], 'b')

plt.subplot(13, 1, 3)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 19], 'b')

plt.subplot(13, 1, 4)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 20], 'b')

plt.subplot(13, 1, 5)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 21], 'b')

plt.subplot(13, 1, 6)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 22], 'b')

plt.subplot(13, 1, 7)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 23], 'b')

plt.subplot(13, 1, 8)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 24], 'b')

plt.subplot(13, 1, 9)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 25], 'b')

plt.subplot(13, 1, 10)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 26], 'b')

plt.subplot(13, 1, 11)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 27], 'b')

plt.subplot(13, 1, 12)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 28], 'b')

plt.subplot(13, 1, 13)
plt.ylim([0, 1.2])
plt.plot(index[:], feat_data[:, 29], 'b')

plt.show()
print(len(feat_data[0]))


class window_builder:
    def __init__(self, en=encoder(), raw=[]):
        self.raw = raw
        self.encoder = en
        self.data = []
        self.label = []
        self.index = []

    def preprocess_default_feature_average(self, data=None):
        if data is not None:
            self.raw = np.asarray(data)

        self.data_dim = self.encoder.emg_dim + self.encoder.flex_dim
        print(f"data_dim : {self.data_dim}")

        if (self.encoder.seq_length >= len(self.raw)):
            print(f"Error : seqence length {self.encoder.seq_length} is shorter than data count {len(self.raw)}")
            return

        dataX = []
        dataY = []
        dataT = []
        for i in range(len(self.raw) - self.encoder.seq_length + 1):
            _x = self.raw[i:i + self.encoder.seq_length,
                 self.encoder.index_dim: self.encoder.index_dim + self.encoder.emg_dim]
            _y = self.raw[i + self.encoder.seq_length - 1, self.encoder.index_dim + self.encoder.emg_dim:]
            _t = self.raw[i:i + self.encoder.seq_length, :self.encoder.index_dim]
            dataX.append(_x)
            dataY.append(_y)
            dataT.append(_t)

        self.index = np.array(dataT)
        self.data = np.array(dataX)
        self.label = np.array(dataY)

        return self.index, self.data, self.label


seq_length = 2 ** 8

EC_WB = encoder(emg_dim=8 * 3, flex_dim=5, seq_length=seq_length)
WB = window_builder(en=EC_WB, raw=feat_data)
trainIndex, trainData, trainLabel = WB.preprocess_default_feature_average(feat_data)

# print(f"Index example: {trainIndex[0].reshape(-1)}\nData example: {trainData[0]} \nLabel example: {trainLabel[0]}\n")
print(f"Data example: {trainData[0]}\nLabel example: {trainLabel[0]}\n")


class network_default_feature_average:
    def __init__(self, data_encoder=encoder()):
        tf.set_random_seed(777)  # reproducibility
        self.data_encoder = data_encoder
        self.seq_length = data_encoder.seq_length
        self.data_dim = data_encoder.emg_dim
        self.output_dim = data_encoder.flex_dim

        self.graph = tf.Graph()
        self.sess = tf.Session()

        self.flag_kernel_opened = False
        self.flag_placeholder = False

    # build a LSTM network
    def build_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim, state_is_tuple=True)
        return cell

    def construct_placeholders(self, learning_rate=0.1, hidden_dim=30, stack_dim=2):
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.stack_dim = stack_dim

        # Input Place holders
        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_encoder.emg_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

        # Build a LSTM network
        multi_cells = tf.contrib.rnn.MultiRNNCell([self.build_cell() for _ in range(self.stack_dim)],
                                                  state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
        self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_dim)

        # Cost & Loss & Optimizer
        self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        # RMSE
        self.targets = tf.placeholder(tf.float32, [None, self.output_dim])
        self.predictions = tf.placeholder(tf.float32, [None, self.output_dim])
        self.rmse = []
        for i in range(self.output_dim):
            rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets[:, i] - self.predictions[:, i])))
            self.rmse.append(rmse)
        # self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

        self.flag_kernel_opened = True
        self.flag_placeholder = True

    def train_network(self, training_data=[], training_label=[], iterations=5000, batch_size=2 ** 14,
                      location='model/_'):
        # self.sess = tf.Session(graph=self.graph)
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
        self.loss_set = []
        self.batch_size = batch_size

        # Training step
        for i in range(iterations):
            batch_loss = 0
            for idx in range(len(training_data) // self.batch_size):
                if idx + self.batch_size < len(training_data):
                    _, step_loss = self.sess.run([self.train, self.loss],
                                                 feed_dict={self.X: training_data[idx:idx + self.batch_size],
                                                            self.Y: training_label[idx:idx + self.batch_size]})
                    batch_loss += step_loss
                else:
                    _, step_loss = self.sess.run([self.train, self.loss],
                                                 feed_dict={self.X: training_data[idx:], self.Y: training_label[idx:]})
                    batch_loss += step_loss

            self.loss_set.append(batch_loss)

            print(f"[iter: {i}/{iterations}] loss: {batch_loss}")

        # train_predict = self.sess.run(self.Y_pred, feed_dict={self.X: training_data})

        # Save Network
        self.saver = tf.train.Saver()
        # location = location + "(" + str(self.stack_dim) + ")"
        self.saver.save(self.sess, location + "/lstm.ckpt")
        return self.loss_set

    def restore(self, location='model/_'):
        if (not self.flag_placeholder):
            self.construct_placeholders()
        if (os.path.exists(location) == False):
            print("Error : No such location")
            return
        self.sess = tf.Session(graph=self.graph)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(location))

    def infer(self, testSet=[], testLabel=None, default_=0.39):
        prediction = []
        for idx in range(len(testSet)):
            test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: [testSet[idx]]})
            prediction.append(test_predict[0])
            if idx % 100 == 0:
                print(f"index : {idx}/{len(testSet)}")

        # Calculate RMSE
        if testLabel is not None:
            rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: testLabel, self.predictions: prediction})
            print(f"RMSE: {rmse_val}\n")
            return prediction, rmse_val
        return prediction

        # Calculate RMSE
        if testLabel is not None:
            rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: testLabel, self.predictions: prediction})
            print(f"RMSE: {rmse_val}\n")
            return prediction, rmse_val
        return prediction

    def close(self):
        tf.reset_default_graph()
        self.sess.close()


DL.load(testing_file)

CR = cropper(en=EC, raw=DL.raw)
CR_data = CR.delete_abnormal_sEMG()

FT_data = FT.filter_highpass(data=CR_data, cutoff=10)

SC.scale(FT_data)
SC_data = SC.raw

FE = feature_extractor(en=EC)
FE_data = FE.feature_average(data=SC_data)

testIndex, testData, testLabel = WB.preprocess_default_feature_average(data=FE_data)

# print(f"Index example: {testIndex[0].reshape(-1)}\nData example: {testData[0]} \nLabel example: {testLabel[0]}\n")
print(f"Data example: {testData[0]} \nLabel example: {testLabel[0]}\n")

subject = subject_name
seq_length = seq_length

learning_rate = 2 ** (-13)
iteration = 2 ** 9
batch_size = 2 ** 10

stack_dim = 2
# stack_dim : the number of layer of LSTM cells
hidden_dim = 100
# hidden_dim : the number of units in the LSTM cell
input_dim = (emg_dim * 3 + flex_dim) * seq_length
output_dim = flex_dim

Network = network_default_feature_average(data_encoder=EC_WB)

with Network.graph.as_default():
    Network.construct_placeholders(learning_rate=learning_rate, hidden_dim=hidden_dim, stack_dim=stack_dim)
    print("Model Constructed\n")
    Network.restore(location=)
    loss = Network.train_network(training_data=trainData, training_label=trainLabel, iterations=iteration,
                                 batch_size=batch_size, location=model_dir)

    print("Train Over\n")
    print("Model " + model_name + " saved\n")

    prediction, rmse_val = Network.infer(testData, testLabel)


class plotter:
    def __init__(self, training_name="", testing_name="", net="", learning_rate=0, iteration=0, seq_length=3,
                 stack_dim=0, hidden_dim=0, rmse=[], prediction=[], label=[], index=[], flex_dim=5):

        self.training_name = training_name
        self.testing_name = testing_name
        self.net = net
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.seq_length = seq_length
        self.stack_dim = stack_dim
        self.hidden_dim = hidden_dim
        self.rmse = rmse
        self.prediction = prediction
        self.label = label
        self.index = index
        self.flex_dim = flex_dim

    def plot_encoded(self, subplot_row=2, size=(20, 10), figloc='./result/tmp'):
        fig = plt.figure(num=5, figsize=size)
        plt.figure(5)
        # print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")

        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0]) + 1) // subplot_row), i + 1)

            if i < (len(self.label[0]) - self.flex_dim):
                plt.ylim([-20000, 20000])
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
            f"Training : {self.training_name}, Testing : {self.testing_name}, Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

    def plot_comparison(self, subplot_row=2, size=(20, 10), figloc='./result/tmp'):
        fig = plt.figure(num=6, figsize=size)
        plt.figure(6)
        # print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")
        for i in range(len(self.label[0])):
            plt.subplot(subplot_row, int((len(self.label[0]) + 1) // subplot_row), i + 1)
            plt.ylim([0, 1])
            plt.xlabel("time(s)")
            plt.plot(self.index[:, :, 0], self.prediction[:, i], '--r', self.index[:, :, 0], self.label[:, i], 'b')
            if i < (len(self.label[0]) - self.flex_dim):
                plt.title(f"emg ch {i + 1},rmse {self.rmse[i]:0.3f}")
            elif i >= (len(self.label[0]) - self.flex_dim):
                plt.title(f"flex order {i + 1 - (len(self.label[0]) - self.flex_dim)},rmse {self.rmse[i]:0.3f}")
        plt.suptitle(
            f"Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

    def plot_rmse(self, subplot_row=2, size=(20, 10), figloc='./result/tmp'):
        fig = plt.figure(num=7, figsize=size)
        plt.figure(7)

        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

    def plot_training_graph(self, loss=[], iteration=5000, size=(20, 10), figloc='./result'):
        step = list(range(iteration))
        min_loss = np.amin(loss)
        fig = plt.figure(num=8, figsize=size)
        plt.figure(8)
        plt.title(f"Model {self.net},min_loss {min_loss}")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.plot(step, loss)
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()


p = plotter(training_name, testing_name, "network_default_feature_average", learning_rate=learning_rate,
            iteration=iteration, seq_length=seq_length, stack_dim=stack_dim, hidden_dim=hidden_dim, rmse=rmse_val,
            prediction=np.asarray(prediction), label=testLabel, index=testIndex, flex_dim=flex_dim)

train_fig_name = now.strftime("%Y%m%d_%H%M_train")
train_fig_dir = fig_dir + '/' + train_fig_name
p.plot_training_graph(loss, iteration, figloc=train_fig_dir)
print("Figure " + train_fig_name + " saved\n")

test_fig_name = now.strftime("%Y%m%d_%H%M_test")
test_fig_dir = fig_dir + '/' + test_fig_name
p.plot_comparison(subplot_row=2, size=(20, 10), figloc=test_fig_dir)
print("Figure " + test_fig_name + " saved\n")
test_data_name = now.strftime("%Y%m%d_%H%M_test.csv")

test_data_dir = fig_dir + '/' + test_data_name
test_data = np.concatenate((testIndex[:, 0], testLabel, np.asarray(prediction)), axis=1)
np.savetxt(test_data_dir, test_data, delimiter=',')
print(f"Data {test_data_name} saved\n")