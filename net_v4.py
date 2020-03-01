import os
import sys
import shutil
import math
import numpy as np
import pandas as pd
import scipy.special
from Bio import SeqIO

from fasta_parser import Parser


class Nnet:

    def __init__(self,
                 data_cd=None,
                 data_nc=None,
                 data_qr=None,
                 layers_num=7,
                 epochs=10000,
                 model=None
                 ):
        self.data_cd = data_cd.drop([0], axis=1).values
        self.data_nc = data_nc.values if data_nc is not None else [0]
        self.data_qr = data_qr.values if data_qr is not None else [0]
        self.labels_cd = data_cd[0].values

        self.data_tr = None
        self.labels_tr = None
        self.model_w = None
        self.model_b = None
        self.feat_num = len(data_qr.columns)
        self.layers_num = layers_num - 2
        self.epochs = epochs
        self.path = model if model is not None else model
        self.out = None
        self.activation = lambda x: scipy.special.expit(x)

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
        self.data_tr = (self.data_cd / 255.0 * 0.99) + 0.01
        ind = list(self.labels_cd)
        targets = np.zeros((len(ind), 10))
        for i in range(len(ind)):
            targets[i, ind[i]] = 0.99
        self.labels_tr = targets

    def set_model(self):
        """model weights & biases
           initialization
        """
        weights = dict()
        weights['layer_0-1'] = np.random.normal(0.0,
                                                pow(self.feat_num, -0.5),
                                                (100, self.feat_num)
                                                )
        for n in range(1, self.layers_num):
            weights['layer_' + str(n) + '-' +  str(n + 1)] = np.random.normal(0.0,
                                                                              pow(100, -0.5),
                                                                              (100, 100)
                                                                              )
        weights['layer_' + str(self.layers_num) + '-' + str(self.layers_num + 1)] = np.random.normal(0.0,
                                                                                                     pow(10, -0.5),
                                                                                                     (10, 100)
                                                                                                     )
        self.model_w = weights

        biases = dict()
        # biases['layer_0'] = np.zeros((self.feat_num,)) + 0.01
        for n in range(1, self.layers_num + 1):
            biases['layer_' + str(n)] = np.zeros((100,)) + 0.01
        biases['layer_' + str(self.layers_num + 1)] = np.zeros((10,)) + 0.01
        self.model_b = biases

    def forward_pass(self, inp):
        """returns net output
           by layers
        """
        forward_dic = dict()
        forward_dic['layer_0'] = inp
        for n in range(1, self.layers_num + 2):
            forward_dic['layer_' + str(n)] = self.activation(np.dot(self.model_w['layer_' + str(n - 1) + '-' + str(n)],
                                                                    forward_dic['layer_' + str(n - 1)]) +
                                                             self.model_b['layer_' + str(n)]
                                                             )
        return forward_dic

    def backward_pass(self, outp, target):
        """returns net errors
           by layers
        """
        backward_dic = dict()
        backward_dic['layer_' + str(self.layers_num + 1)] = target - outp['layer_' + str(self.layers_num + 1)]
        for n in range(self.layers_num, -1, -1):
            backward_dic['layer_' + str(n)] = np.dot(self.model_w['layer_' + str(n) + '-' + str(n + 1)].T,
                                                     backward_dic['layer_' + str(n + 1)])
        # for k, v in backward_dic.items():
        #     print(k, v.shape)
        # exit(0)
        return backward_dic

    def update(self, outputs, errors, lr):
        for n in range(self.layers_num + 1, 0, -1):
            a = errors['layer_' + str(n)] * outputs['layer_' + str(n)] * (1 - outputs['layer_' + str(n)])
            a = a.reshape(len(a), 1)
            b = outputs['layer_' + str(n - 1)]
            b = b.reshape(1, len(b))
            self.model_w['layer_' + str(n - 1) + '-' + str(n)] += lr * np.dot(a, b)
            self.model_b['layer_' + str(n)] += errors['layer_' + str(n)]

    def train(self):
        # batches = Nnet.get_batches(self.data_tr, 100)
        # self.data_tr = None
        for e in range(self.epochs):
            # np.random.shuffle(self.data_tr)
            print('\n-Processing epochs: {} of {}...'.format(e + 1, self.epochs))
            lr = self.lr_schedule(e)
            # batch = batches[np.random.randint(0, len(batches) - 1)]
            n = 0
            l = len(self.data_tr)
            for m in range(l):
                n += 1
                done = int(50 * n / l)
                sys.stdout.write("\r[%s%s] %s" % ('=' * done, ' ' * (50 - done),
                                 str(n) + '/' + str(l) + ' ' + str(2 * done) + '% complete'))
                sys.stdout.flush()

                outputs = self.forward_pass(self.data_tr[m, :])
                errors = self.backward_pass(outputs, self.labels_tr[m])
                self.update(outputs, errors, lr)
        print('\n-Training complete')

    def update_constants(self):
        pass

    def load_model(self):
        if not os.path.isdir('model'):
            print('Error: cannot load model, folder "model" does not exist')
            exit(1)
        else:
            print('-Loading saved model...')
            files = os.listdir('model/')
            weights = dict()
            biases = dict()
            for f in files:
                if 'w_layer' in f:
                    name = f[2:-4]
                    weights[name] = np.load('model/' + f)
                if 'b_layer' in f:
                    name = f[2:-4]
                    biases[name] = np.load('model/' + f)
            if len(weights) == len(biases):
                self.model_w = weights
                self.model_b = biases
                print('-Complete')
            else:
                print('Error: cannot build model, invalid parameters')
                exit(1)

    def save_model(self):
        if os.path.isdir('model'):
            shutil.rmtree('model', ignore_errors=True)
        os.mkdir('model')
        print('-Saving model...')
        for key, value in self.model_w.items():
            np.save('model/w_' + key, value)
        for key, value in self.model_b.items():
            np.save('model/b_' + key, value)
        print('-Saved')

    def predict(self):
        predictions = []
        print('-Predicting...')
        for n in range(len(self.data_qr)):
            pred = self.forward_pass(self.data_qr[n])
            pred = pred['layer_' + str(self.layers_num + 1)]
            predictions.append(np.argmax(pred))
        print('-Complete')
        return predictions

    def lr_schedule(self, epoch):
        """returns a custom learning rate
           that decreases as epochs progress.
        """
        epochs = self.epochs
        learning_rate = 0.7
        if epoch > epochs * 0.5:
            learning_rate = 0.5
        if epoch > epochs * 0.75:
            learning_rate = 0.01
        if epoch > epochs * 0.9:
            learning_rate = 0.007
        return learning_rate

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

    @classmethod
    def crossval_csv(cls, fasta_cd, fasta_nc, threads=1):
        accurs = []
        if os.path.isdir('crossval'):
            shutil.rmtree('crossval', ignore_errors=True)
        os.mkdir('crossval')

        # Parsing ref coding file
        parser_cd = Parser(fasta_cd)
        parser_cd.parse()
        cd_feat = parser_cd.gen_feat_tab()
        cd_rscu = parser_cd.rscu_tab(save=False)
        # cd_hex_freq = parser_cd.gen_hex_tab()
        # hex_in_cd = parser_cd.count_hex()

        # Parsing ref noncoding file
        parser_nc = Parser(fasta_nc)
        parser_nc.parse()
        nc_feat = parser_nc.gen_feat_tab()
        nc_rscu = parser_nc.rscu_tab(save=False)
        # nc_hex_freq = parser_nc.gen_hex_tab()
        # hex_in_nc = parser_nc.count_hex()

        used_cd, used_nc = [], []
        for n in range(1, 11):
            print('-Splitting files ' + str(n) + '/10...')
            locdir = 'crossval/fold_' + str(n) + '/'
            os.mkdir(locdir)
            len_cd = len(cd_feat)
            len_nc = len(nc_feat)
            test_len = int(len_nc * 0.1) if len_nc <= len_cd else int(len_cd * 0.1)

            # forming cd files
            test_records_cd, train_records_cd = [], []
            nums_cd = [m for m in range(len(cd_feat))]
            test_nums_cd, train_nums_cd = [], []
            while len(test_nums_cd) < test_len:
                ind = nums_cd.pop(np.random.randint(0, len(nums_cd)))
                if ind not in used_cd and ind not in test_nums_cd:
                    test_nums_cd.append(ind)
                    used_cd.append(ind)
                if len(nums_cd) == 0:
                    break
            for k in range(len(cd_feat)):
                if k in test_nums_cd:
                    test_records_cd.append(cd_feat.iloc[k])
                else:
                    train_records_cd.append(cd_feat.iloc[k])

            test_cd_out = pd.concat(test_records_cd, axis=1).T
            train_cd_out = pd.concat(train_records_cd, axis=1).T
            del test_records_cd, train_records_cd
            train_cd_out.to_csv(locdir + 'train_cd.csv', sep=';')

            # forming nc files
            test_records_nc, train_records_nc = [], []
            nums_nc = [m for m in range(len(nc_feat))]
            test_nums_nc, train_nums_nc = [], []
            while len(test_nums_nc) < test_len:
                ind = nums_nc.pop(np.random.randint(0, len(nums_nc)))
                if ind not in used_nc and ind not in test_nums_nc:
                    test_nums_nc.append(ind)
                    used_nc.append(ind)
                if len(nums_nc) == 0:
                    break
            for k in range(len(nc_feat)):
                if k in test_nums_nc:
                    test_records_nc.append(nc_feat.iloc[k])
                else:
                    train_records_nc.append(nc_feat.iloc[k])

            test_nc_out = pd.concat(test_records_nc, axis=1).T
            train_nc_out = pd.concat(train_records_nc, axis=1).T
            del test_records_nc, train_records_nc
            train_nc_out.to_csv(locdir + 'train_nc.csv', sep=';')

            # merding test files
            test_out = pd.concat([test_cd_out, test_nc_out], axis=0, ignore_index=True)
            del test_nc_out, test_cd_out
            test_out.to_csv(locdir + 'test.csv', sep=';')
            del test_out, train_cd_out, train_nc_out
        del used_cd, used_nc

        for n in range(1, 11):
            print('-Processing folds ' + str(n) + '/10...')
            locdir = 'crossval/fold_' + str(n) + '/'

            train_cd = pd.read_csv(locdir + 'train_cd.csv', sep=';')
            train_nc = pd.read_csv(locdir + 'train_cd.csv', sep=';')
            test = pd.read_csv(locdir + 'test.csv', sep=';')
            query_names = test['Name']

            nnet = Nnet(data_cd=train_cd.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0),
                        data_nc=train_nc.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0),
                        data_qr=test.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0),
                        layers_num=7,
                        epochs=7
                        )
            del train_nc, train_cd, test
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
            del res, query_names
            accurs.append(Nnet.accuracy(locdir + "prediction.txt", 'Pp', 'CNT'))
        print(accurs)
        return accurs

    @classmethod
    def crossval_fasta(cls, fasta_cd, fasta_nc, threads=1):
        accurs = []
        if os.path.isdir('crossval'):
            shutil.rmtree('crossval', ignore_errors=True)
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
            parser_cd = Parser(locdir + 'train_cd.fasta')
            parser_cd.parse()
            cd_feat = parser_cd.gen_feat_tab()
            cd_rscu = parser_cd.rscu_tab(save=False)
            cd_hex_freq = parser_cd.gen_hex_tab()
            hex_in_cd = parser_cd.count_hex()

            # Parsing ref noncoding file
            parser_nc = Parser(locdir + 'train_nc.fasta')
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
                        layers_num=7,
                        epochs=10
                        )
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
        print(accurs)
        return accurs

    @classmethod
    def accuracy(cls, file, prefix_cd, prefix_nc):
        """counts accuracy of the prediction
           using prediction file and prefixes of
           coding and noncoding transcripts.
        """
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
