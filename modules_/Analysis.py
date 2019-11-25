import numpy as np
import matplotlib.pyplot as plt
import os

"""
  Plot Graph with Sensor(Flex) and Prediction Values for comparison
"""

class plotter:
    def __init__(self, prediction = [], label = [], index = [], dim = 6):
        self.prediction = prediction
        self.label = label
        self.index = index
        self.dim = dim
    
    def plot_comparison(self, subplot_row = 2, size = (20,10), figloc = './result/tmp'):
        plt.figure(2)
        fig = plt.figure(figsize=size)

        print(f"{self.index[:3, :, 0].reshape(-1)} \n{self.prediction[:3, 0]} \n{self.label[:3, 0]}")

        for i in range(self.dim):
            plt.subplot(subplot_row, int( self.dim/subplot_row ), i+1)
            plt.ylim([-1000,1000])
            plt.plot(self.index[:,:,0], self.prediction[:,i],'r', self.index[:,:,0], self.label[:,i],'b')
        fig.savefig(figloc, dpi=fig.dpi)