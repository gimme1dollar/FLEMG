from _modules import Encoder,Processor
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
from drawnow import drawnow

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

# TODO
class realtime_plotter(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self._exit = False
        self.iter = 0

    def draw(self):
        """
        xList.append(iter)
        xNum = len(xList)
        xLen = 100

        for i in range(Enco.flex_dim):
            flex_data[i].append(testLabel[-1][i])
            flex_pred[i].append(prediction[0][i])
        for i in range(Enco.emg_dim):
            emg_data[i].append(testData[-1][-1][i])

        plt.subplot(2, 1, 1)
        plt.axis([np.clip(xNum, 0, xNum - xLen), xNum - 1, 0, 500])
        for i in range(len(flex_data)):
            plt.plot(xList, flex_data[i], '+')
        for i in range(len(emg_data)):
            plt.plot(xList, emg_data[i], 'o')

        plt.subplot(2, 1, 2)
        plt.axis([np.clip(xNum, 0, xNum - xLen), xNum - 1, 0, 1000])
        for i in range(len(flex_pred)):
            plt.plot(xList, flex_pred[i], '+')
        """

    def run(self):
        while True:
            drawnow(self.draw)
            plt.pause(0.001)
            self.iter += 1

            if self._exit :
                break

    def exit(self):
        self._exit = True
