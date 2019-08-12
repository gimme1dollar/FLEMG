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
    
    def plot_comparison(self, subplot_row = 2, size = (20,10)):
        plt.figure(2)
        plt.figure(figsize=size)

        for i in range(self.dim):
            plt.subplot(subplot_row, int( self.dim/subplot_row ), i+1)
            plt.ylim([0,0.5])
            plt.plot(self.index[:,:,0], self.prediction[:,i],'r', self.index[:,:,0], self.label[:,i],'b')