import numpy as np
import queue
import time
import struct

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

        self.tmp_I = 0
        self.tmp_E = []
        self.tmp_F = []

    def encode_FLEX(self, idx = None):
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
            self.tmp_F = []
            return

        self.tmp_F = f_d
        return

    def encode_EMG_GUI(self):
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
            print("E")
        except:
            self.tmp_E = []
            return

        self.tmp_E = e_d
        return

    def encode_EMG_Serial(self):
        e_d = []
        try:
            e_q = self.queue_list[1].queue
            e_str = list(e_q)[-1] #b'\x00\xc0\xa0\x05\xaa\n,\xabX\xac\xaf+)\xae\x82g\xb3\x9d\xc9\xb3>\x06\xa9\xec<\xabt\xd2\x00\x00\x00\x00\x00\x00\xc0\xa0\x06\xa9\xfat\xaa\x9c+\xae\x8f;\xad\xce\x03\xb2\xef\xe5\xb3m<\xa9\xdd\xfc\xadb \x000\x08\x90'

            e_list = str(e_str).split("\\x") # ["b'", '85', 'bc', 'e6', 'e5', 'bc', 'd9', 'be', 'bc', 'e8', 'da', 'bd$?', 'c0`', 'a4', 'c1', '92', 'b8', 'c1', '01u', 'c0', 'ca', 'e7', '00', '00', '00', '00', '00', '00', 'c0', 'a0', '86', 'bc', 'f2', 'f9', 'bc', 'e5', 'ca', 'bc', 'f5"', 'bd', 'b2', '84', 'c0', 'de', 'e9', 'c1L', 'a1', 'bf', 'aa', 'c1', 'be', 'feV', '00', '00', '00', '00', '00', '00', "c0'"]

        except:
            self.tmp_E = []
            return

        for i in range(len(e_list)):
            if e_list[i].find("a0") > -1 :
                print(str(e_list[i]))
                """
                e_d.append(e_str[i + 1])  # Ch1
                e_d.append()  # Ch2
                e_d.append()  # Ch3
                e_d.append()  # Ch4
                e_d.append()  # Ch5
                e_d.append()  # Ch6
                e_d.append()  # Ch7
                e_d.append()  # Ch8
                """
                break


        self.tmp_E = e_d
        return

    def encode_IDX(self, idx):
        if idx is not None:
            tmp = [idx]
        else:
            tmp = [self.count]

        self.encode_EMG_Serial()
        self.encode_FLEX()

        #Store
        if( self.tmp_F == self.flex_dim and self.tmp_E == self.emg_dim ):
            tmp += self.tmp_E
            tmp += self.tmp_F
            self.dataSet.append(tmp) # [2320, 345, 407, 413, 412, 392, -576, -582, -529, -523, 617, 656, 631, -475]
            self.count += 1

        for i in range( len(self.queue_list) ):
            self.queue_list[i].queue.clear()

        return