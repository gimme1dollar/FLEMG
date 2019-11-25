import numpy as np
import queue

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

        #FLEX : b'352,403,423,399,384\n352,402,423,399,384\n352,402,423,399,384\n352,402,423,399,384\n352,402,423,399,384\n352,402,423,399,384\n352,402,423,399,384\n352,'
        f_d = []
        f_q = self.queue_list[0].queue
        if len(f_q) > 0: # serial로부터 받아온 내용이 없을 수도 있음
            f_str = str(list(f_q)[-1]) # b'411,389\n341,407,412,411,389\n341,407,412,411,389\n341,407,412,411,389\n341,406,412,411,389\n341,407,412,411,389\n341,406,412,411,389\n341,407,412,411,389\n341,407,412,411,389\n341,407,412,411,389\n341,'

            f_token = [] # would-be :: ['3', '4', '1', ',', '4', '0', '6', ',', '4', '1', '2', ',', '4', '1', '1', ',', '3', '8', '9']
            for i in range( len(f_str) ):
                char = f_str[-(i+1)]
                if char == "\\":
                    start = 1
                    while True:
                        try:
                            char = f_str[-(i + 1 + start)]
                            if char == "n":
                                break
                            f_token.insert(0, char)
                            start += 1
                        except IndexError:
                            break
                    break
            f_str = str(f_token).replace('\'', '').replace(', ', '') # [341,406,412,411,389]

            f_list = f_str.split(",")
            if (len(f_list) == self.flex_dim):
                f_d.append(int(f_list[0].split('[')[1]))
                for i in range(self.flex_dim - 2):
                    f_d.append(int(f_list[i + 1]))
                f_d.append(int(f_list[self.flex_dim - 1].split(']')[0]))

        #EMG : b'.045,0.997,1.061,-1.106][-2.031,-1.946,-1.973,-2.073,1.959,1.865,2.013,-2.003][-2.354,-2.301,-2.370,-2.330,2.225,2.118,2.278,-2.307][-1.331,-1.308,-1.388,-1.233,1.111,1.040,1.122,-1.239][0.535'
        e_d = []
        e_q = self.queue_list[1].queue
        if len(e_q) > 0:  # serial로부터 받아온 내용이 없을 수도 있음
            e_str = str(list(e_q)[-1]) # b'784][4.286,4.266,4.221,4.248,-4.392,-4.448,-4.440,4.283][3.850,3.759,3.731,3.818,-3.948,-4.019,-4.011,3.834][2.193,2.106,2.059,2.134,-2.150,-2.208,-2.185,2.169][1.453,1.405,1.341,1.370,-1.290,'

            e_token = []
            for i in range(len(e_str)):
                char = e_str[-(i + 1)]
                if char == "]":
                    start = 1
                    while True:
                        try:
                            char = e_str[-(i + 1 + start)]
                            if char == "[":
                                break
                            e_token.insert(0, char)
                            start += 1
                        except IndexError:
                            break
                    break
            e_str = str(e_token).replace('\'', '').replace(', ', '') # [-0.615,-0.541,-0.589,-0.550,0.452,0.565,0.554,-0.614]

            e_list = e_str.split(",")
            if (len(e_list) == self.emg_dim):
                e_d.append(int(float(e_list[0].split('[')[1]) * 1000))
                for i in range(self.emg_dim - 2):
                    e_d.append(int(float(e_list[i + 1]) * 1000))
                e_d.append(int(float(e_list[self.emg_dim - 1].split(']')[0]) * 1000))

        #Store
        if( len(f_d) == self.flex_dim and len(e_d) == self.emg_dim ):
            tmp += f_d
            tmp += e_d
            self.dataSet.append(tmp) # [2320, 345, 407, 413, 412, 392, -576, -582, -529, -523, 617, 656, 631, -475]
            self.count += 1

        for i in range( len(self.queue_list) ):
            self.queue_list[i].queue.clear()