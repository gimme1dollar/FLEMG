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

    def encode_FLEX_readline(self):
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

    def encode_FLEX_read(self):
        f_d = []
        try:
            f_q = self.queue_list[0].queue
            f_str = str(list(f_q)[-1]) # b'396,415,410,422\n305,396,415,410,422\n304,396,415,41'

            f_raw = f_str.split("\\n")
            f_list = f_raw[1].split(",") #['304', '396', '414', '405', '412']
            if (len(f_list) == self.flex_dim):
                f_d.append(int(f_list[0]))
                for i in range(self.flex_dim - 2):
                    f_d.append(int(f_list[i + 1]))
                f_d.append(int(f_list[self.flex_dim - 1].split('\\')[0]))

            print(f"f_d {len(f_d)}")
            self.tmp_F = f_d
        except:
            return
        return

    def encode_EMG_line(self):
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
            self.tmp_E = []
            return

        self.tmp_E = e_d
        return

    def encode_EMG_bytearray(self):
        e_d = []
        try:
            e_q = self.queue_list[1].queue
            e_str = list(e_q)[-1] #b"\xc0;\x1a\x00\x00\x00\x00\x00\x00\xc0\xa0\x82\xbf\x06*\xbe\xcb\xe4\xbf \x99\xbe\xe2G\xbf\xb7\x82\xc0\x15'\xbf\xe2\xbe\xc07#\x00\x00\x00\x00\x00\x00\xc0\xa0\x83\xbf\xaa/\xbfm\xc0\xbf\xc5h\xbf\x85L\xbe\x93l\xc0\r>\xbf\x97"
            pivot = e_str.find(160) # 0xa0 is 160

            ch1 = e_str[pivot + 2:pivot + 5] #ch1
            ch2 = e_str[pivot + 5:pivot + 8]  # ch2
            ch3 = e_str[pivot + 8:pivot + 11]  # ch3
            ch4 = e_str[pivot + 11:pivot + 14]  # ch4
            ch5 = e_str[pivot + 14:pivot + 17]  # ch5
            ch6 = e_str[pivot + 17:pivot + 20]  # ch6
            ch7 = e_str[pivot + 20:pivot + 23]  # ch7
            ch8 = e_str[pivot + 23:pivot + 26]  # ch8

            e_d.append(int("{0:b}".format(ch1[0]) + "{0:b}".format(ch1[1]) + "{0:b}".format(ch1[2])))
            e_d.append(int("{0:b}".format(ch2[0]) + "{0:b}".format(ch2[1]) + "{0:b}".format(ch2[2])))
            e_d.append(int("{0:b}".format(ch3[0]) + "{0:b}".format(ch3[1]) + "{0:b}".format(ch3[2])))
            e_d.append(int("{0:b}".format(ch4[0]) + "{0:b}".format(ch4[1]) + "{0:b}".format(ch4[2])))
            e_d.append(int("{0:b}".format(ch5[0]) + "{0:b}".format(ch5[1]) + "{0:b}".format(ch5[2])))
            e_d.append(int("{0:b}".format(ch6[0]) + "{0:b}".format(ch6[1]) + "{0:b}".format(ch6[2])))
            e_d.append(int("{0:b}".format(ch7[0]) + "{0:b}".format(ch7[1]) + "{0:b}".format(ch7[2])))
            e_d.append(int("{0:b}".format(ch8[0]) + "{0:b}".format(ch8[1]) + "{0:b}".format(ch8[2])))

            self.tmp_E = e_d
        except:
            return
        return

    def encode_IDX(self, idx):
        if idx is not None:
            tmp = [idx]
        else:
            tmp = [self.count]

        self.encode_FLEX_read()
        self.encode_EMG_bytearray()

        print(f"F : {len(self.tmp_F)}")
        print(f"E : {len(self.tmp_E)}")

        #Store
        if( self.tmp_F == self.flex_dim and self.tmp_E == self.emg_dim ):
            tmp += self.tmp_E
            tmp += self.tmp_F
            self.dataSet.append(tmp) # [2320, 345, 407, 413, 412, 392, -576, -582, -529, -523, 617, 656, 631, -475]
            print(self.dataSet)
            self.count += 1

        for i in range( len(self.queue_list) ):
            self.queue_list[i].queue.clear()

        return