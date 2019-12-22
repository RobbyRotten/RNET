from net_v4 import Nnet
import numpy as np
import pandas as pd
# n = Nnet()
# n.set_model()

path_cd = "mnist_dataset/mnist_train_100.csv"
coding = pd.read_csv(path_cd, sep=',', header=None)

path_qr = "mnist_dataset/mnist_test_10.csv"
query = pd.read_csv(path_qr, sep=',', header=None)
true_lab = query[0]
query = query.drop([0], axis=1)

nnet = Nnet(data_cd=coding, data_qr=query, layers_num=7, epochs=1)
nnet.set_model()
nnet.preprocessing()
nnet.train()
# nnet.save_model()
# nnet.load_model()

'''
labels_out = nnet.predict()

res = ''
for n in range(len(labels_out)):
    res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
with open("prediction_np.txt", 'w') as f_obj:
    f_obj.write(res)
'''
