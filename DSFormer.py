import numpy as np
from torch.utils.data import Dataset
# import pandas as pd


class CustomSet(Dataset):
    def __init__(self, source):
        self.samples = source

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def __features__(self):
        return self.samples[0][0].shape[0]


class DSFormer:
    def __init__(self,
                 data_cd=None,
                 data_nc=None,
                 data_qr=None
                 ):

        self.data_cd = data_cd.values if data_cd is not None else [0]
        self.data_nc = data_nc.values if data_nc is not None else [0]
        self.data_qr = data_qr.values if data_qr is not None else [0]
        self.labels_cd = np.array([0.99 for n in range(len(self.data_cd))])
        self.labels_nc = np.array([0.001 for n in range(len(self.data_nc))])
        self.feat_num = len(data_cd.columns)
        self.data_tr = None
        self.labels_tr = None
        self.values_tr = None

        # Normalization constants
        self.maximums = [69.14414414414415,
                         9951,
                         7278,
                         0.9991554054054054,
                         195.9333333333333,
                         0.0845505,
                         0.631579,
                         0.6754383732308656,
                         0.93100413323957,
                         0.3309859154929577,
                         17.535421439956426,
                         2.252834841995392,
                         ]
        self.means = [46.78307216803418,
                      1973.155838454785,
                      970.9385425812115,
                      0.4772441903932988,
                      43.83702684809195,
                      0.02883090006145734,
                      0.3753700581211571,
                      0.12742843999343276,
                      0.8830413127997834,
                      0.03742352601370512,
                      17.115641742511894,
                      0.2246882655730706,
                      ]

    def normalize(self, arr):
        columns = []
        for n in range(arr.shape[1]):
            normalized = arr[:, n]
            normalized -= self.means[n]
            normalized /= self.maximums[n]
            columns.append(normalized)
        normalized = np.hstack(columns)
        normalized = normalized.reshape(self.feat_num, len(arr)).T
        return normalized

    def preprocessing(self):
        if self.data_cd is not None and self.data_nc is not None:
            self.data_cd = self.normalize(self.data_cd)
            self.data_nc = self.normalize(self.data_nc)
        if self.data_qr is not None:
            self.data_qr = self.normalize(self.data_qr)

        self.data_tr = np.concatenate((self.data_cd, self.data_nc), axis=0)
        self.labels_tr = np.concatenate((self.labels_cd, self.labels_nc))

        self.data_cd, self.data_nc = None, None
        self.labels_cd, self.labels_nc = None, None

        self.values_tr = []
        for n in range(len(self.data_tr)):
            self.values_tr.append((self.data_tr[n],
                                   self.labels_tr[n]))

        """
        self.data_tr = np.hstack([self.data_tr, self.labels_tr.reshape(self.labels_tr.shape[0], 1)])
        np.random.shuffle(self.data_tr)
        self.values_tr = self.data_tr[:, :self.feat_num].reshape(len(self.data_tr), self.feat_num)
        self.labels_tr = self.data_tr[:, self.feat_num].reshape(len(self.data_tr), 1)
        """

    def get_train(self):
        """
        train_values = CustomSet(self.values_tr)
        train_labels = CustomSet(self.labels_tr)
        return train_values, train_labels
        """
        train_values = CustomSet(self.values_tr)
        return train_values

    def get_query(self):
        query = CustomSet(self.data_qr)
        return query
