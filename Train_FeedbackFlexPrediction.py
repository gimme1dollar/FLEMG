import os
from os import path
#from google.colab import drive
from datetime import datetime
import numpy as np
#import pandas as pd
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
notebook_path_name = '.'

subject_name = "YHyu"
network_name = "Network_feedback_flex_prediction_feature_average"
time_name = now.strftime("%Y%m%d_%H%M")
model_name = notebook_path_name + '/model/' + network_name


try:
# Create target Directory
    os.makedirs(model_name)
    print("Directory " , model_name ,  " Created\n")
    subject_dir = model_name + '/' + subject_name
    try:
        os.makedirs(subject_dir)
        print("Directory" , subject_dir , "Created\n")
    except FileExistsError:
        print("Directory " , subject_dir ,  " already exists\n")
except FileExistsError:
    subject_dir = model_name + '/' + subject_name
    try:
        os.makedirs(subject_dir)
        print("Directory" , subject_dir , "Created\n")
    except FileExistsError:
        print("Directory " , subject_dir ,  " already exists\n")

model_dir = subject_dir + '/' +time_name


fig_name =  notebook_path_name + "/result/" + network_name
try:
# Create target Directory
    os.makedirs(fig_name)
    print("Directory " , fig_name ,  " Created\n")
    subject_dir = fig_name + "/" + subject_name
    try:
        os.makedirs(subject_dir)
        print("Directory",subject_dir,"Created\n")
    except FileExistsError:
        print("Directory",subject_dir,"already exists\n")

except FileExistsError:
    subject_dir = fig_name + "/" + subject_name
    try:
        os.makedirs(subject_dir)
        print("Directory",subject_dir,"Created\n")
    except FileExistsError:
        print("Directory",subject_dir,"already exists\n")
fig_dir = subject_dir





#### Data Loading

subject_dir = notebook_path_name + '/data/' + subject_name
target_dir = subject_dir
training_name = "20200129_2127_feature"
testing_name = "20200129_2201_feature"
training_file = target_dir + "/"+training_name + ".csv"
testing_file = target_dir + "/"+testing_name + ".csv"

class encoder:
    def __init__(self, queue_list = [], index_dim = 1, flex_dim = 5, emg_dim = 8, seq_length = 3, emg_active_dim = 8):
        self.queue_list = queue_list
        self.dataSet = []
        self.count = 0

        self.emg_dim = emg_dim
        self.flex_dim = flex_dim
        self.index_dim = index_dim
        self.data_dim =  emg_dim+flex_dim
        self.label_dim = flex_dim
        self.seq_length = seq_length
        self.emg_active_dim = emg_active_dim

        self.tmp_I = 0
        self.tmp_E = []
        self.tmp_F = []

class data_loader:
    def __init__(self, en = encoder(), raw = []):
        self.raw = raw
        self.encoder = en
        self.data = []
        self.label = []
        self.index = []

    def load(self, location='default', delimiter = ','):
        self.raw = np.loadtxt(location, delimiter = delimiter)

emg_dim = 8
flex_dim = 5
EC = encoder(emg_dim = emg_dim ,flex_dim = flex_dim)

DL = data_loader(en = EC)
DL.load(training_file)
raw_data = DL.raw



class window_builder:
    def __init__(self, en=encoder(), raw=[]):
        self.raw = raw
        self.encoder = en
        self.data = []
        self.label = []
        self.index = []

    def preprocess_feedback_flex_feature_average(self, data=None):
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
                 self.encoder.index_dim: self.encoder.index_dim + self.encoder.emg_dim + self.encoder.flex_dim]
            _y = self.raw[i + self.encoder.seq_length - 1,
                 self.encoder.index_dim + self.encoder.emg_dim:self.encoder.index_dim + self.encoder.emg_dim + self.encoder.flex_dim]
            _t = self.raw[i:i + self.encoder.seq_length,
                 0:self.encoder.index_dim]
            dataX.append(_x)
            dataY.append(_y)
            dataT.append(_t)

            if i % 100 == 0:
                print(f"{i}/{len(self.raw) - self.encoder.seq_length + 1} Window Building...")

        self.index = np.array(dataT)
        self.data = np.array(dataX)
        self.label = np.array(dataY)

        return self.index, self.data, self.label


seq_length = 2**8

EC_WB = encoder(emg_dim = 8 * 3, flex_dim = 5, seq_length = seq_length)
WB = window_builder(en = EC_WB, raw=raw_data)

trainIndex, trainData, trainLabel = WB.preprocess_feedback_flex_feature_average()
#print(f"Index example: {trainIndex[0].reshape(-1)}\nData example: {trainData[0]} \nLabel example: {trainLabel[0]}\n")
print(f"Training Data example: {trainData[0]}\nLabel example: {trainLabel[0]}\n")


DL.load(testing_file)
DL_data = DL.raw
testIndex, testData, testLabel = WB.preprocess_feedback_flex_feature_average(data=DL_data)
#print(f"Index example: {testIndex[0].reshape(-1)}\nData example: {testData[0]} \nLabel example: {testLabel[0]}\n")
print(f"Test Data example: {testData[0]} \nLabel example: {testLabel[0]}\n")





### Network



class network_feedback_flex_prediction_feature_average:
    def __init__(self, data_encoder = encoder()):
        tf.set_random_seed(777)  # reproducibility
        self.data_encoder = data_encoder
        self.seq_length = data_encoder.seq_length
        self.data_dim = data_encoder.emg_dim + data_encoder.flex_dim
        self.output_dim = data_encoder.flex_dim

        self.graph = tf.Graph()
        self.sess = tf.Session()

        self.flag_kernel_opened = False
        self.flag_placeholder = False

    # build a LSTM network
    def build_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_dim, state_is_tuple=True)
        return cell

    def construct_placeholders(self, learning_rate = 0.1, hidden_dim = 30, stack_dim = 2):
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.stack_dim = stack_dim

        # Input Place holders
        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.output_dim])

        # Build a LSTM network
        multi_cells = tf.contrib.rnn.MultiRNNCell([self.build_cell() for _ in range(self.stack_dim)], state_is_tuple=True)
        outputs, _states=tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
        self.Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_dim)

        # Cost & Loss & Optimizer
        self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        # RMSE
        self.targets = tf.placeholder(tf.float32, [None, self.output_dim])
        self.predictions = tf.placeholder(tf.float32, [None, self.output_dim])
        self.rmse=[]
        for i in range(self.output_dim):
            rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets[:,i] - self.predictions[:,i])))
            self.rmse.append(rmse)
        #self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

        self.flag_kernel_opened = True
        self.flag_placeholder = True

    def train_network_prediction(self, training_data = [], training_label = [], iterations = 5000, batch_size = 2**14, training_data_changing_iter_size = 100, save_checkpoint=2 ** 5, location = 'model/_', restore = False):
        if restore is False:
            self.sess = tf.Session(graph=self.graph)
            init = tf.global_variables_initializer()
            self.sess.run(init)

        self.loss_set=[]
        self.batch_size = batch_size
        self.saver = tf.train.Saver()

        # Training step
        for i in range(iterations):
            batch_loss = 0

            idx = 0
            for idx in range( len(training_data)//self.batch_size ):
                _, step_loss = self.sess.run([self.train, self.loss], feed_dict={self.X: training_data[(idx) * batch_size:(idx + 1) * batch_size], self.Y: training_label[(idx) * batch_size:(idx + 1) * batch_size]})
                batch_loss += step_loss
            _, step_loss = self.sess.run([self.train, self.loss], feed_dict={self.X: training_data[(idx + 1) * batch_size:], self.Y: training_label[(idx + 1) * batch_size:]})
            batch_loss += step_loss

            self.loss_set.append(batch_loss)

            print(f"[iter: {i}/{iterations}] loss: {batch_loss}")


            # Alter training_data flex true values to flex predictions
            if i % training_data_changing_iter_size == 0 and i is not 0:
                prediction = self.infer_feedback(testSet = training_data)
                for i in range(len(training_data)-seq_length):
                    if i > seq_length:
                        training_data[i+seq_length, :, -self.data_encoder.flex_dim:] = prediction[i:i+seq_length]

            # Save Network
            if i % save_checkpoint == 0 and i is not 0:
                self.saver.save(self.sess, location+"/lstm.ckpt", i)
                np.savetxt( location+"/loss_"+str(i)+".txt", self.loss_set, delimiter=',')
                print(f"[iter: {i}/{iterations}] Model saved at {location}")

        self.saver.save(self.sess, location + "/lstm.ckpt", i)
        print(f"[iter: {i}/{iterations}] Model saved at {location}")

        return self.loss_set

    def restore(self, location='model/_'):
        if (not self.flag_placeholder) :
            self.construct_placeholders()
        if (os.path.exists(location) == False):
            print("Error : No such location")
            return
        self.sess = tf.Session(graph=self.graph)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(location))


    def infer_feedback(self, testSet=[], testLabel=None, default_=0.39):
        prediction = []
        for idx in range(len(testSet)):
            # testSet Reconstruction
            if idx == 0:
                for j in range(self.data_encoder.label_dim):
                    for l in range(self.data_encoder.seq_length):
                        pass
                    # testSet[idx, l, self.data_encoder.emg_dim + j] = default_
                    # testSet[idx, l, self.data_encoder.emg_dim + j] = random.random()
            else:
                for j in range(self.data_encoder.label_dim):
                    for l in range(self.data_encoder.seq_length):
                        if l == self.data_encoder.seq_length - 1:
                            testSet[idx, l, self.data_encoder.emg_dim + j] = test_predict[0][j]
                        else:
                            testSet[idx, l, self.data_encoder.emg_dim + j] = testSet[
                                idx - 1, l + 1, self.data_encoder.emg_dim + j]

            # Feed testData
            test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: [testSet[idx]]})

            # Stack Prediction
            if idx == 0:
                prediction = test_predict
            else:
                prediction = np.vstack((prediction, test_predict))

            if idx % 5000 == 0:
                print(f"{idx}/{len(testSet)} pred : {test_predict[0]}")

        # Calculate RMSE
        if testLabel is not None:
            rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: testLabel, self.predictions: prediction})
            print(f"RMSE: {rmse_val}\n")
            return prediction, rmse_val
        return prediction


    def infer_batch(self, testSet = [], testLabel = None, batch_size = 2**9, default_ = 0.39):
        prediction = []

        # Inference with batch
        if len(testSet) < batch_size :
            prediction = self.sess.run(self.Y_pred, feed_dict={self.X: testSet[:]})
        else :
            idx = 0
            for idx in range( len(testSet)//batch_size ):
                test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: testSet[(idx)*batch_size:(idx+1)*batch_size]})

                if idx == 0:
                    prediction = test_predict
                else:
                    prediction = np.vstack((prediction, test_predict))
            test_predict = self.sess.run(self.Y_pred, feed_dict={self.X: testSet[(idx+1)*batch_size:]})
            prediction = np.vstack((prediction, test_predict))

        # Calculate RMSE
        print("Calculating RMSE")
        if testLabel is not None:
            rmse_val = self.sess.run(self.rmse, feed_dict={self.targets: testLabel, self.predictions: prediction})
            print(f"RMSE: {rmse_val}\n")
            return prediction, rmse_val
        return prediction


    def close(self):
        tf.reset_default_graph()
        self.sess.close()

















subject = subject_name
seq_length = seq_length

learning_rate = 2**(-10)
iteration = 2**11
batch_size = 2**10
training_data_changing_iter_size = 2**7
save_checkpoint = 2**5

stack_dim = 2   #stack_dim : the number of layer of LSTM cells
hidden_dim = 300   #hidden_dim : the number of units in the LSTM cell

Network = network_feedback_flex_prediction_feature_average(data_encoder=EC_WB)





with Network.graph.as_default():
    print("Model Constructing\n")
    Network.construct_placeholders(learning_rate=learning_rate, hidden_dim=hidden_dim, stack_dim=stack_dim)

    print("Model Restoring\n")
    #Network.restore(location=model_name+"/YHyu/20200202_1907")

    print("Model Training\n")
    loss=Network.train_network_prediction(training_data = trainData, training_label = trainLabel,iterations = iteration, batch_size = batch_size, save_checkpoint= save_checkpoint, training_data_changing_iter_size = training_data_changing_iter_size, location = model_dir, restore=False)

    print("Model " + model_name + " saved\n")

    prediction, rmse_val = Network.infer(testData, testLabel)



















class plotter:
    def __init__(self, training_name="",testing_name="",net = "", learning_rate = 0, iteration = 0 ,seq_length = 3 ,stack_dim = 0 ,hidden_dim = 0, rmse = [], prediction = [], label = [], index = [], flex_dim = 5):

        self.training_name=training_name
        self.testing_name=testing_name
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
        fig = plt.figure(num=5,figsize=size)
        plt.figure(5)
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
        plt.suptitle(f"Training : {self.training_name}, Testing : {self.testing_name}, Model : {self.net}, Alpha : {self.learning_rate}, Iteration : {self.iteration}, Seq_length : {self.seq_length}, Stack_dim : {self.stack_dim}, Hidden_dim : {self.hidden_dim}, avgRMSE : {np.mean(self.rmse):0.3f}")
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

    def plot_comparison(self, subplot_row = 2, size = (20,10), figloc = './result/tmp'):
        fig = plt.figure(num=6,figsize=size)
        plt.figure(6)
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
        fig = plt.figure(num=7,figsize=size)
        plt.figure(7)

        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

    def plot_training_graph(self, loss=[], iteration=5000, size=(20,10), figloc = './result'):
        step=list(range(iteration))
        min_loss = np.amin(loss)
        fig = plt.figure(num=8,figsize=size)
        plt.figure(8)
        plt.title(f"Model {self.net},min_loss {min_loss}")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.plot(step,loss)
        fig.savefig(figloc, dpi=fig.dpi)
        plt.show()

p = plotter(training_name,testing_name,"network_default_feature_average",learning_rate=learning_rate,iteration=iteration, seq_length=seq_length, stack_dim=stack_dim, hidden_dim=hidden_dim, rmse=rmse_val,prediction=np.asarray(prediction),label=testLabel, index=testIndex, flex_dim=flex_dim)

test_data_name = now.strftime("%Y%m%d_%H%M_test.csv")
test_data_dir = fig_dir + '/' + test_data_name
test_data=np.concatenate((testIndex[:,0],testLabel,np.asarray(prediction)),axis=1)
np.savetxt(test_data_dir,test_data,delimiter=',')
print(f"Data {test_data_name} saved\n")

train_fig_name = now.strftime("%Y%m%d_%H%M_train")
train_fig_dir = fig_dir + '/' + train_fig_name
p.plot_training_graph(loss,iteration,figloc=train_fig_dir)
print("Figure "+train_fig_name+" saved\n")

train_data_name = now.strftime("%Y%m%d_%H%M_loss.csv")
train_data_dir = fig_dir + '/' + train_data_name
train_data=loss
np.savetxt(train_data_dir,train_data,delimiter=',')
print(f"Data {train_data_name} saved\n")

test_fig_name = now.strftime("%Y%m%d_%H%M_test")
test_fig_dir = fig_dir + '/' + test_fig_name
p.plot_comparison(subplot_row = 2, size = (20,10), figloc=test_fig_dir)
print("Figure "+test_fig_name+" saved\n")
