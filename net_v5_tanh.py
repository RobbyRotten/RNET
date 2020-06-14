import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from conda import history


class Nnet:

    def __init__(self,
                 data_cd=None,
                 data_nc=None,
                 data_qr=None,
                 layers_num=7,
                 epochs=10000,
                 hidden=100,
                 model=None,
                 threads=1,
                 lr=0.7
                 ):
        self.data_cd = data_cd.values if data_cd is not None else [0]
        self.data_nc = data_nc.values if data_nc is not None else [0]
        self.data_qr = data_qr.values if data_qr is not None else [0]
        self.labels_cd = np.array([0.99 for n in range(len(self.data_cd))])
        self.labels_nc = np.array([0.001 for n in range(len(self.data_nc))])
        self.data_tr = None
        self.labels_tr = None
        self.model = None
        self.feat_num = len(data_cd.columns)
        self.layers_num = layers_num - 2
        self.epochs = epochs
        self.path = model if model is not None else model
        self.out = None
        self.threads = threads
        self.hidden = hidden
        self.learning_rate = lr

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

    def preprocessing(self):
        if self.data_cd is not None and self.data_nc is not None:
            self.data_cd = self.normalize(self.data_cd)
            self.data_nc = self.normalize(self.data_nc)
        if self.data_qr is not None:
            self.data_qr = self.normalize(self.data_qr)

    def normalize(self, arr):
        columns = []
        for n in range(arr.shape[1]):
            normalized = arr[:, n]
            normalized -= self.means[n]
            normalized /= self.maximums[n]
            columns.append(normalized)
        normalized = np.hstack(columns)
        normalized = normalized.reshape(arr.shape[1], len(arr)).T
        return normalized

    def set_model(self):

        self.data_tr = np.concatenate((self.data_cd, self.data_nc), axis=0)
        self.labels_tr = np.concatenate((self.labels_cd, self.labels_nc))

        self.data_cd, self.data_nc = None, None
        self.labels_cd, self.labels_nc = None, None

        self.data_tr = np.hstack([self.data_tr, self.labels_tr.reshape(self.labels_tr.shape[0], 1)])
        np.random.shuffle(self.data_tr)

        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(self.hidden,
                                    activation=tf.keras.activations.tanh,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=0),
                                    bias_initializer=tf.keras.initializers.RandomNormal(seed=0),
                                    input_shape=(self.feat_num, )))
        stored_hidden = self.hidden
        for n in range(self.layers_num):
            stored_hidden = int(stored_hidden * 0.5)
            self.model.add(layers.Dense(stored_hidden,
                                        activation=tf.keras.activations.tanh,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=0),
                                        bias_initializer=tf.keras.initializers.RandomNormal(seed=0),
                                        )
                           )
        self.model.add(layers.Dense(1,
                                    activation=tf.keras.activations.tanh,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=0),
                                    bias_initializer=tf.keras.initializers.RandomNormal(seed=0),
                                    ))

        self.model.compile(optimizer=tf.keras.optimizers.SGD(self.learning_rate),
                           loss=tf.keras.metrics.mean_squared_error,
                           metrics=[tf.keras.metrics.mean_squared_error]
                           # [tf.keras.metrics.mean_absolute_error]
                           )
    # gridsearch scicit learn - подбор параметров - class
    # torch nn module ->
    # relu, scaled exp lu
    # batch normalization
    # к-во слоев, нелинейность, оптимизатор, регуляризация, lr
    # tf v2 / pytorch

    def lr_schedule(self, epoch):
        """returns a custom learning rate
           that decreases as epochs progress.
        """
        epochs = self.epochs
        learning_rate = 0.5
        if epoch > epochs * 0.5:
            learning_rate = 0.25
        if epoch > epochs * 0.75:
            learning_rate = 0.01
        if epoch > epochs * 0.9:
            learning_rate = 0.007
        return learning_rate

    def train(self):
        # lr_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
        history = self.model.fit(self.data_tr[:, :self.feat_num].reshape(len(self.data_tr), self.feat_num),
                                 self.data_tr[:, self.feat_num].reshape(len(self.data_tr), 1),
                                 epochs=self.epochs,
                                 batch_size=100
                                 # callbacks=lr_callback
                                 )
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def update_constants(self):
        pass

    def load_model(self):
        path = 'model/model' if self.path is None else self.path
        self.model.load_weights(path)

    def save_model(self):
        try:
            os.mkdir('model')
        except FileExistsError:
            shutil.rmtree('model')
            os.mkdir('model')
        finally:
            self.model.save_weights('model/model')

    def predict(self):
        labels_pr = self.model.predict(self.data_qr)
        labels_out = list(label[0] for label in labels_pr)
        return labels_out

    @classmethod
    def get_batches(cls, arr, size):
        np.random.shuffle(arr)
        divider = int(len(arr) / size)
        last_ind = divider * size
        batches = np.vsplit(arr[:last_ind], divider)
        last_batch = arr[last_ind:]
        if len(last_batch) != 0:
            batches.append(last_batch)
        return batches
