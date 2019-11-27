import numpy as np
import queue
import time

class encoder:
    def __init__(self, queue_list = [], index_dim = 1, flex_dim = 5, emg_dim = 8, seq_length = 3):
        self.queue_list = queue_list
        self.dataSet = []
        self.count = 0
        
        self.emg_dim = emg_dim
        self.flex_dim = flex_dim
        self.index_dim = index_dim
        self.data_dim =  emg_dim + flex_dim
        self.label_dim = flex_dim
        self.seq_length = seq_length

    def encode(self, idx = None):
        if idx is not None:
            tmp = [idx]
        else:
            tmp = [self.count]

        #FLEX
        f_d = []
        try:
            f_q = self.queue_list[0].queue
            f_str = str(list(f_q)[-1]) # b'341,407,412,411,389\n

            f_list = f_str.split(",")
            if (len(f_list) == self.flex_dim):
                f_d.append(int(f_list[0].split('\'')[1]))
                for i in range(self.flex_dim - 2):
                    f_d.append(int(f_list[i + 1]))
                f_d.append(int(f_list[self.flex_dim - 1].split('\\')[0]))
        except:
            return

        #EMG
        e_d = []
        try:
            e_q = self.queue_list[1].queue
            e_str = str(list(e_q)[-1])

            e_list = e_str.split(",")
            if (len(e_list) == self.emg_dim):
                e_d.append(float(e_list[0].split('\'')[1]))
                for i in range(self.emg_dim - 2):
                    e_d.append(float(e_list[i + 1]))
                e_d.append(float(e_list[self.emg_dim - 1].split('\\')[0]))
        except:
            return

        #Store
        if( len(f_d) == self.flex_dim and len(e_d) == self.emg_dim ):
            tmp += e_d
            tmp += f_d
            self.dataSet.append(tmp) # [2320, 345, 407, 413, 412, 392, -576, -582, -529, -523, 617, 656, 631, -475]
            self.count += 1

        for i in range( len(self.queue_list) ):
            self.queue_list[i].queue.clear()

        return