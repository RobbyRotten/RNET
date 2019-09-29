import numpy
import pandas as pd
import sys

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import to_categorical

classes = 1
feat_num = 72  # 72 or 12
lab_num = 73


def data_extr(fr):
    cols = list(fr[n] for n in fr.columns[:feat_num + 1])
    acc_cols = []
    for col in cols:
        if col.max() >= 1:
            acc_col = col / (col.max() * 1.05)
            acc_cols.append(acc_col)
        else:
            acc_cols.append(col)
    # selecting columns with biggest difference
    data = pd.DataFrame([acc_cols[1], acc_cols[3],
                         acc_cols[5], acc_cols[11]]).T
    # data = pd.DataFrame(acc_cols).T
    data = data.values
    # data = data * 10
    return data


def labels_extr_two(fr, classes=classes):
    labels_l = []
    for n in range(len(fr)):
        num = fr.iloc[n][73]
        label = numpy.zeros(classes) + 0.0001
        label[int(num - 1)] = 0.99
        labels_l.append(label)
    labels = numpy.vstack(labels_l).astype(numpy.float64)
    return labels


def labels_extr_one(fr):
    labels_l = []
    for n in range(len(fr)):
        num = fr.iloc[n][lab_num]
        # 1 - coding, 2 - noncoding
        if num == 1:
            label = 0.99
        else:
            label = 0.01
        labels_l.append(label)
    labels = numpy.array(labels_l).astype(numpy.float64)
    return labels


training_file = 'CSV/Features-training_all-open.csv'
training_data_list = pd.read_csv(training_file, header=None, sep=',',
                                 low_memory=False).drop(0).astype(numpy.float64)
data_tr = data_extr(training_data_list)
labels_tr = labels_extr_one(training_data_list)

"""
test_file = 'resultTabNew.csv'
test_data_list = pd.read_csv(test_file, header=None, sep=',',
                         low_memory=False).drop(0).astype(numpy.float64)
data_te = data_extr(test_data_list)
labels_te = labels_extr_one(test_data_list)
"""

# data_out = pd.concat([pd.DataFrame(data_tr), pd.DataFrame(labels_tr)], axis=1).reset_index(drop=True)
# data_out.to_csv('training_norm_all72.csv')

# mn_cod = data_tr[:9891].mean(axis=0)
# mn_nonc = data_tr[9891:].mean(axis=0)
# mn = numpy.vstack([mn_cod, mn_nonc])
# sub = mn_cod - mn_nonc
# print(sub, len(sub)) # [ 0.30490397  1.07962346 -0.70335604  1.92616031  0.9289795   1.05714589
#		      0.8034433   0.01181701  0.67876622  0.00923576 -0.01850306  1.39563599]

# max difference between indexes 1, 3, 5, 11 + 4?
# 1 - ORF length, 3 - ORF coverage, 5 - GC, 11 - SCUO (Synonymous Codon Usage Order)
# 4 - TranscriptLength	

model = tf.keras.Sequential()
model.add(layers.Dense(feat_num, activation='sigmoid'))
model.add(layers.Dense(100, activation='sigmoid', bias_regularizer=tf.keras.regularizers.l2(0.001),
                       kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(layers.Dense(1000, activation='sigmoid', bias_regularizer=tf.keras.regularizers.l2(0.001),
                       kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(layers.Dense(1000, activation='sigmoid', bias_regularizer=tf.keras.regularizers.l2(0.001),
                       kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(layers.Dense(100, activation='sigmoid', bias_regularizer=tf.keras.regularizers.l2(0.001),
                       kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(layers.Dense(10, activation='sigmoid', bias_regularizer=tf.keras.regularizers.l2(0.001),
                       kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(layers.Dense(classes, activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),  # tf.train.GradientDescentOptimizer(0.0001),
              loss='mse',  # 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data_tr, labels_tr, epochs=10)  # ,batch_size=100)
labels_pr = model.predict(data_tr)
labels_out = list(label[0] for label in labels_pr)

with open('prediction_RTNew_TF_selected.txt', 'w') as f:
    f.write(str(labels_out))
