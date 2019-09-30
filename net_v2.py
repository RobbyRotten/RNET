import os
import shutil
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from Bio import SeqIO

from fasta_parser import Parser


class Nnet:

    def __init__(self, data_cd=None, data_nc=None, data_qr=None, layers=7, epochs=10000, model=None):
        self.data_cd = data_cd.values if data_cd is not None else [0]
        self.data_nc = data_nc.values if data_nc is not None else [0]
        self.data_qr = data_qr.values if data_qr is not None else [0]
        self.labels_cd = np.array([0.99 for n in range(len(self.data_cd))])
        self.labels_nc = np.array([0.001 for n in range(len(self.data_nc))])
        self.data_tr = None
        self.labels_tr = None
        self.model = None
        self.feat_num = len(data_qr.columns) if data_qr is not None else len(data_cd.columns)
        self.layers_num = layers - 2
        self.epochs = epochs
        self.path = model if model is not None else model

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
        normalized = normalized.reshape(self.feat_num, len(arr)).T
        return normalized

    def set_model(self):
        self.data_tr = np.concatenate((self.data_cd, self.data_nc), axis=0)
        self.labels_tr = np.concatenate((self.labels_cd, self.labels_nc))

        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(100, activation='sigmoid', input_shape=(self.feat_num, )))
        for n in range(self.layers_num):
            self.model.add(layers.Dense(100,
                                        activation='sigmoid',
                                        kernel_initializer=tf.keras.initializers.random_normal(seed=0),
                                        bias_initializer=tf.keras.initializers.random_normal(seed=0)
                                        )
                           )
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(optimizer=tf.train.GradientDescentOptimizer(0.7),  # tf.train.AdamOptimizer(0.01),
                           loss=tf.keras.losses.mean_squared_error,           # 'msle',  # 'categorical_crossentropy',
                           metrics=[tf.keras.metrics.mean_absolute_error]     # ['accuracy'])
                           )

    def train(self):
        self.model.fit(self.data_tr, self.labels_tr, epochs=self.epochs, batch_size=100)

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
    def crossval_fasta(cls, fasta_cd, fasta_nc):
        accurs = []
        try:
            os.mkdir('crossval')
        except FileExistsError:
            shutil.rmtree('crossval')
            os.mkdir('crossval')
        records_cd, records_nc = [], []

        # forming cd files
        for record in SeqIO.parse(fasta_cd, "fasta"):
            records_cd.append(record)
        used_cd, used_nc = [], []
        test_len_cd = int(len(records_cd) * 0.1)
        for n in range(1, 11):
            print('-Splitting coding file ' + str(n) + '/10...')
            locdir = "crossval/fold_" + str(n) + '/'
            os.mkdir(locdir)
            test_records_cd, train_records_cd = [], []
            nums_cd = [m for m in range(len(records_cd))]
            test_nums_cd, train_nums_cd = [], []
            while len(test_nums_cd) < test_len_cd:
                ind = nums_cd.pop(np.random.randint(0, len(nums_cd)))
                if ind not in used_cd and ind not in test_nums_cd:
                    test_nums_cd.append(ind)
                    used_cd.append(ind)
                if len(nums_cd) == 0:
                    break
            for k in range(len(records_cd)):
                if k in test_nums_cd:
                    test_records_cd.append(records_cd[k])
                else:
                    train_records_cd.append(records_cd[k])
            test_cd_out, train_cd_out = '', ''
            for record in test_records_cd:
                test_cd_out += '>' + str(record.name) + '\n' + str(record.seq) + '\n'
            with open(locdir + 'test_cd.fasta', 'w') as f_obj:
                f_obj.write(test_cd_out)
            del test_records_cd, test_cd_out
            for record in train_records_cd:
                train_cd_out += '>' + str(record.name) + '\n' + str(record.seq) + '\n'
            with open(locdir + 'train_cd.fasta', 'w') as f_obj:
                f_obj.write(train_cd_out)
            del train_records_cd, train_cd_out
        del records_cd

        # forming nc files
        for record in SeqIO.parse(fasta_nc, "fasta"):
            records_nc.append(record)
        test_len_nc = int(len(records_nc) * 0.1)
        test_nums_nc, train_nums_nc = [], []
        for n in range(1, 11):
            print('-Splitting noncoding file ' + str(n) + '/10...')
            locdir = "crossval/fold_" + str(n) + '/'
            test_records_nc, train_records_nc = [], []
            nums_nc = [m for m in range(len(records_nc))]
            while len(test_nums_nc) < test_len_nc:
                ind = nums_nc.pop(np.random.randint(0, len(nums_nc)))
                if ind not in used_nc and ind not in test_nums_nc:
                    test_nums_nc.append(ind)
                    used_nc.append(ind)
                if len(nums_nc) == 0:
                    break
            for k in range(len(records_nc)):
                if k in test_nums_nc:
                    test_records_nc.append(records_nc[k])
                else:
                    train_records_nc.append(records_nc[k])
            test_nc_out, train_nc_out = '', ''
            for record in test_records_nc:
                test_nc_out += '>' + str(record.name) + '\n' + str(record.seq) + '\n'
            with open(locdir + 'test_nc.fasta', 'w') as f_obj:
                f_obj.write(test_nc_out)
            del test_records_nc, test_nc_out
            for record in train_records_nc:
                train_nc_out += '>' + str(record.name) + '\n' + str(record.seq) + '\n'
            with open(locdir + 'train_nc.fasta', 'w') as f_obj:
                f_obj.write(train_nc_out)
            del train_records_nc, train_nc_out
        del records_nc

        # merging test files
        for n in range(1, 11):
            locdir = "crossval/fold_" + str(n) + '/'
            with open(locdir + 'test_nc.fasta', 'r') as f_obj:
                nc = f_obj.read()
            with open(locdir + 'test_cd.fasta', 'r') as f_obj:
                cd = f_obj.read()
            with open(locdir + 'test_nc_cd.fasta', 'w') as f_obj:
                f_obj.write(nc + cd[:-1])
            del nc, cd

        # training and predicting
        for n in range(1, 11):
            print('-Processing folds ' + str(n) + '/10...')
            locdir = "crossval/fold_" + str(n) + '/'

            # Parsing test file
            parser_qr = Parser(locdir + 'test_nc_cd.fasta')
            parser_qr.parse()
            qr_feat = parser_qr.gen_feat_tab()
            qr_rscu = parser_qr.rscu_tab(save=False)
            hex_in_qr = parser_qr.count_hex()

            # Parsing ref coding file
            parser_cd = Parser(locdir + 'test_cd.fasta')
            parser_cd.parse()
            cd_feat = parser_cd.gen_feat_tab()
            cd_rscu = parser_cd.rscu_tab(save=False)
            cd_hex_freq = parser_cd.gen_hex_tab()
            hex_in_cd = parser_cd.count_hex()

            # Parsing ref noncoding file
            parser_nc = Parser(locdir + 'test_nc.fasta')
            parser_nc.parse()
            nc_feat = parser_nc.gen_feat_tab()
            nc_rscu = parser_nc.rscu_tab(save=False)
            nc_hex_freq = parser_nc.gen_hex_tab()
            hex_in_nc = parser_nc.count_hex()

            h_score_nc = Nnet.hex_score(cd_hex_freq, nc_hex_freq, hex_in_nc)
            h_score_cd = Nnet.hex_score(cd_hex_freq, nc_hex_freq, hex_in_cd)
            h_score_qr = Nnet.hex_score(cd_hex_freq, nc_hex_freq, hex_in_qr)
            h_sc_nc_df = pd.DataFrame([nc_feat['Name'], h_score_nc]).T
            h_sc_cd_df = pd.DataFrame([cd_feat['Name'], h_score_cd]).T
            h_sc_qr_df = pd.DataFrame([qr_feat['Name'], h_score_qr]).T
            del nc_hex_freq, cd_hex_freq, hex_in_nc, hex_in_cd, hex_in_qr
            del h_score_cd, h_score_nc, h_score_qr

            for df in [h_sc_nc_df, h_sc_cd_df, h_sc_qr_df]:
                df.columns = ['Name', 'Hex_score']
            cd_feat = pd.merge(cd_feat, h_sc_cd_df, on='Name')
            nc_feat = pd.merge(nc_feat, h_sc_nc_df, on='Name')
            qr_feat = pd.merge(qr_feat, h_sc_qr_df, on='Name')
            query_names = qr_feat.Name
            del h_sc_cd_df, h_sc_nc_df, h_sc_qr_df

            nnet = Nnet(data_cd=cd_feat.drop(['Name'], axis=1).fillna(0),
                        data_nc=nc_feat.drop(['Name'], axis=1).fillna(0),
                        data_qr=qr_feat.drop(['Name'], axis=1).fillna(0),
                        layers=7,
                        epochs=10)
            del cd_feat, nc_feat, qr_feat, qr_rscu, nc_rscu, cd_rscu
            nnet.preprocessing()
            nnet.set_model()
            nnet.train()
            labels_out = nnet.predict()
            del nnet

            res = ''
            for l in range(len(labels_out)):
                res += query_names[l] + ' ' + str(labels_out[l]) + '\n'
            with open(locdir + "prediction.txt", 'w') as f_obj:
                f_obj.write(res)
            del res
            accurs.append(Nnet.accuracy(locdir + "prediction.txt", 'Pp', 'CNT'))
        return accurs

    @classmethod
    def accuracy(cls, file, prefix_cd, prefix_nc):
        positive = 0
        with open(file, 'r') as f_obj:
            lines = f_obj.readlines()
        all_lines = len(lines)
        for line in lines:
            line = line.split()
            name = line[0]
            pred = float(line[1])
            if prefix_cd in name and pred >= 0.5:
                positive += 1
            elif prefix_nc in name and pred < 0.5:
                positive += 1
        return positive / all_lines

    @classmethod
    def hex_score(cls, ref_cd, ref_nc, query):
        """counts hexamer score for every
           sequence in the query list
           using reference frequencies.
        """
        if str(type(ref_cd)) == "<class 'pandas.core.frame.DataFrame'>" and \
                str(type(ref_nc)) == "<class 'pandas.core.frame.DataFrame'>":
            scores = []
            for seq in query:
                score = 0
                observ_hex = len(query)
                for hexamer in seq:
                    if int(ref_cd[hexamer]) == 0:
                        score += -1
                    elif int(ref_nc[hexamer]) == 0:
                        score += 1
                    else:
                        score += math.log10(int(ref_cd[hexamer]) / int(ref_nc[hexamer]))
                if observ_hex != 0:
                    score /= observ_hex
                scores.append(score)
            return pd.Series(scores)
        else:
            scores = []
            for seq in query:
                score = 0
                observ_hex = len(query)
                for hexamer in seq:
                    if int(ref_cd[hexamer]) == 0:
                        score += -1
                    elif int(ref_nc[hexamer]) == 0:
                        score += 1
                    else:
                        score += math.log10(int(ref_cd[hexamer]) / int(ref_nc[hexamer]))
                if observ_hex != 0:
                    score /= observ_hex
                scores.append(score)
            return pd.Series(scores)
