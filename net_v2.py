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

    def __init__(self,
                 data_cd=None,
                 data_nc=None,
                 data_qr=None,
                 layers_num=7,
                 epochs=10000,
                 model=None,
                 threads=1
                 ):
        self.data_cd = data_cd.values if data_cd is not None else [0]
        self.data_nc = data_nc.values if data_nc is not None else [0]
        self.data_qr = data_qr.values if data_qr is not None else [0]
        self.labels_cd = np.array([0.99 for n in range(len(self.data_cd))])
        self.labels_nc = np.array([0.001 for n in range(len(self.data_nc))])
        self.data_tr = None
        self.labels_tr = None
        self.model = None
        self.feat_num = len(data_qr.columns) if data_qr is not None else len(data_cd.columns)
        self.layers_num = layers_num - 2
        self.epochs = epochs
        self.path = model if model is not None else model
        self.out = None
        self.threads = threads

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
        if os.path.isdir('tensorboard1'):
            shutil.rmtree('tensorboard1', ignore_errors=True)
        os.mkdir('tensorboard1')
        os.mkdir('tensorboard1/metrics')

        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=self.threads,
                                          inter_op_parallelism_threads=self.threads,
                                          allow_soft_placement=True,
                                          device_count={'CPU': self.threads})

        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        os.environ["OMP_NUM_THREADS"] = str(self.threads)
        os.environ["KMP_BLOCKTIME"] = "30"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

        file_writer = tf.compat.v1.summary.FileWriter("tensorboard1/metrics", sess.graph, session=sess)
        # file_writer = tf.contrib.summary.create_file_writer("tensorboard/metrics")
        # file_writer.set_as_default()
        file_writer.add_run_metadata(tf.compat.v1.RunMetadata(), tag='run meta')

        self.data_tr = np.concatenate((self.data_cd, self.data_nc), axis=0)
        self.labels_tr = np.concatenate((self.labels_cd, self.labels_nc))

        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(100, activation='sigmoid', input_shape=(self.feat_num, )))
        for n in range(self.layers_num):
            self.model.add(layers.Dense(100,
                                        activation='sigmoid',
                                        kernel_initializer=tf.compat.v1.keras.initializers.random_normal(seed=0),
                                        bias_initializer=tf.compat.v1.keras.initializers.random_normal(seed=0),
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(),
                                        bias_regularizer=tf.keras.regularizers.l1_l2()
                                        )
                           )
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(optimizer=tf.keras.optimizers.SGD(),
                           loss=tf.keras.losses.mean_squared_error,
                           metrics=['accuracy']                               # [tf.keras.metrics.mean_absolute_error]
                           )

    def model_by_layers(self, start_train=True, predict=True):
        def layer(inp, chan_in, chan_out, name='FC'):
            with tf.name_scope(name):
                w = tf.Variable(tf.random.truncated_normal([chan_in, chan_out], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[1, chan_out]), name='B')
                act = tf.nn.sigmoid(tf.matmul(inp, w) + b)
                tf.compat.v1.summary.histogram('biases', b)
                tf.compat.v1.summary.histogram('weights', w)
                return act

        if os.path.isdir('tensorboard'):
            shutil.rmtree('tensorboard', ignore_errors=True)
        os.mkdir('tensorboard')

        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=self.threads,
                                          inter_op_parallelism_threads=self.threads,
                                          allow_soft_placement=True,
                                          device_count={'CPU': self.threads})

        sess = tf.compat.v1.Session(config=config)
        merged_summary = tf.compat.v1.summary.merge_all()
        writer = tf.compat.v1.summary.FileWriter('tensorboard')
        writer.add_graph(sess.graph)

        os.environ["OMP_NUM_THREADS"] = str(self.threads)
        os.environ["KMP_BLOCKTIME"] = "30"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

        self.data_tr = np.concatenate((self.data_cd, self.data_nc), axis=0)
        self.labels_tr = np.concatenate((self.labels_cd, self.labels_nc))

        x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.feat_num])
        y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        x_inp = tf.reshape(x, [1, self.feat_num])

        layers_dict = dict()
        layers_dict['layer_1'] = layer(x_inp, self.feat_num, 100)
        for n in range(self.layers_num):
            layers_dict['layer_' + str(n+2)] = layer(layers_dict['layer_' + str(n+1)], 100, 100)
        layers_dict['layer_' + str(self.layers_num+2)] = layer(layers_dict['layer_' + str(self.layers_num+1)],
                                                               100, 1)

        layer_out = layers_dict['layer_' + str(self.layers_num+2)]
        mse = tf.reduce_mean(tf.compat.v2.losses.mean_squared_error(y, layer_out))
        lr_placeholder = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate')
        train_step = tf.compat.v1.train.GradientDescentOptimizer(lr_placeholder).minimize(mse)
        weight_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                            mode='FAN_AVG',
                                                                            uniform=True,
                                                                            seed=0)
        bias_initializer = tf.initializers.constant(0.1)
        # correct_pred = tf.equal(layer_out, y)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        sess.run(tf.compat.v1.global_variables_initializer())
        data = np.hstack([self.data_tr, self.labels_tr.reshape(self.labels_tr.shape[0], 1)])
        # print(data.shape, self.feat_num)
        # dtr = data[:, :self.feat_num]

        if start_train:
            e_stored = 1
            for e in range(self.epochs):
                lr = self.lr_schedule(e)
                print("-Processing epoch " + str(e+1) + '/' + str(self.epochs) + '...')
                batches = Nnet.get_batches(data, 100)
                for batch in batches:
                    for j in range(len(batch)):
                        [train_accuracy] = sess.run([mse],
                                                    feed_dict={x: batch[j, :self.feat_num].reshape(1, 12),
                                                               y: batch[j, self.feat_num].reshape(1, 1),
                                                               lr_placeholder: lr
                                                               }
                                                    )
                        if e_stored != e:
                            print('\tmean squared error: ' + str(train_accuracy))
                            # if e % 500 == 0:
                            #    s = sess.run(merged_summary,
                            #                 feed_dict={x: batch[j, :self.feat_num].reshape(1, 12),
                            #                            y: batch[j, self.feat_num].reshape(1, 1),
                            #                            lr_placeholder: lr
                            #                            }
                            #                 )
                            #    writer.add_summary(s, e)
                        sess.run([train_step],
                                 feed_dict={x: batch[j, :self.feat_num].reshape(1, 12),
                                            y: batch[j, self.feat_num].reshape(1, 1),
                                            lr_placeholder: lr
                                            }
                                 )
                        e_stored = e

        """
        if e == self.epochs - 1:
            tf.compat.v1.summary.histogram('input', layers_dict['layer_1'])
            for l in range(self.layers_num):
                lname = 'layer_' + str(l+1)
                tf.compat.v1.summary.histogram(lname, layers_dict[lname])
        """

        out = []
        if predict:
            for l in range(len(self.data_qr)):
                pred = sess.run([layer_out],
                                feed_dict={x: self.data_qr[l].reshape(1, 12)}
                                )
                out.append(pred[0])
        self.out = out

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

        tf.compat.v1.summary.scalar('learning rate', learning_rate)
        return learning_rate

    def train(self):
        lr_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_schedule)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      min_delta=0,
                                                      patience=int(self.epochs * 0.1),
                                                      restore_best_weights=True
                                                      )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard1',
                                                              histogram_freq=1,
                                                              write_grads=True
                                                              )

        self.model.fit(self.data_tr,
                       self.labels_tr,
                       epochs=self.epochs,
                       batch_size=100,
                       callbacks=[tensorboard_callback,
                                  lr_callback
                                  # early_stop
                                  ]
                       )

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
                        epochs=7,
                        threads=threads
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
            del res
            accurs.append(Nnet.accuracy(locdir + "prediction.txt", 'Pp', 'CNT'))
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
                        epochs=10,
                        threads=threads
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
