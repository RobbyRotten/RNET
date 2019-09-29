import sklearn
from sklearn import preprocessing
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import to_categorical


class Nnet:

    def __init__(self, data_cd, data_nc, data_qr, layers=5):
        self.data_cd = data_cd.values
        self.data_nc = data_nc.values
        self.data_qr = data_qr.values
        self.labels_cd = np.array([0.99 for n in range(len(self.data_cd))])
        self.labels_nc = np.array([0.001 for n in range(len(self.data_nc))])
        self.data_tr = None
        self.labels_tr = None
        self.model = None
        self.feat_num = len(data_cd.columns)
        self.layers_num = layers

    def preprocessing(self):
        self.data_cd = self.normalize(self.data_cd)
        self.data_nc = self.normalize(self.data_nc)
        self.data_qr = self.normalize(self.data_qr)

    def normalize(self, arr):
        length = arr.shape[0]
        columns = []
        for n in range(arr.shape[1]):
            column = arr[:, n]
            normalized_value = preprocessing.normalize(column.reshape(1, -1), norm='l2')
            columns.append(normalized_value)
        normalized = np.hstack(columns)
        normalized = normalized.reshape(self.feat_num, len(arr)).T
        return normalized

    def train(self):
        self.data_tr = np.concatenate((self.data_cd, self.data_nc), axis=0)
        print(self.data_tr.shape)
        self.labels_tr = np.concatenate((self.labels_cd, self.labels_nc))
        print(self.labels_tr.shape)

        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(100, activation='sigmoid', input_shape=(self.feat_num, )))
        self.model.add(layers.Dense(100, activation='sigmoid'))  # , bias_regularizer=tf.keras.regularizers.l2(0.01),
                                    # kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        self.model.add(layers.Dense(100, activation='sigmoid'))  # , bias_regularizer=tf.keras.regularizers.l2(0.01),
                                    # kernel_regularizer=tf.keras.regularizers.l2(0.01)))


        self.model.add(layers.Dense(100, activation='sigmoid'))  # , bias_regularizer=tf.keras.regularizers.l2(0.01),
                                    # kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        self.model.add(layers.Dense(100, activation='sigmoid'))  # , bias_regularizer=tf.keras.regularizers.l2(0.01),
                                    # kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        self.model.add(layers.Dense(100, activation='sigmoid'))  # , bias_regularizer=tf.keras.regularizers.l2(0.01),
                                    # kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(optimizer=tf.train.GradientDescentOptimizer(0.7),  # tf.train.AdamOptimizer(0.01),
                           loss=tf.keras.losses.mean_squared_error,  # 'msle',  # 'categorical_crossentropy',
                           metrics=[tf.keras.metrics.mean_absolute_error])  # ['accuracy'])

        self.model.fit(self.data_tr, self.labels_tr, epochs=1000, batch_size=1000)

    def load_model(self):
        pass

    def predict_1(self):
        print(self.data_nc[:10])
        labels_pr = self.model.predict(self.data_nc[:100])
        labels_out = list(label[0] for label in labels_pr)
        return labels_out

    def predict_2(self):
        print(self.data_cd[:10])
        labels_pr = self.model.predict(self.data_cd[:100])
        labels_out = list(label[0] for label in labels_pr)
        return labels_out
