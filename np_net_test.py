from net_v3 import Nnet
import numpy as np
import pandas as pd
# n = Nnet()
# n.set_model()

path_cd = "../RNET1/CSV/CD/Ppatens_prot_all_features_train.csv"
coding = pd.read_csv(path_cd, sep=';')
coding = coding.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0)

path_nc = "../RNET1/CSV/NC/Physcomitrella_patens_lncrnas_features_train.csv"
noncoding = pd.read_csv(path_nc, sep=';')
noncoding = noncoding.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0)

path_qr = "../RNET1/Ppatens_cd_nc_test.csv"
query = pd.read_csv(path_qr, sep=';')
query_names = query.Name
query = query.drop(['Name'], axis=1).fillna(0)

nnet = Nnet(data_cd=coding, data_nc=noncoding, data_qr=query, layers_num=7, epochs=2)
nnet.set_model()
nnet.preprocessing()
nnet.train()
labels_out = nnet.predict()

res = ''
for n in range(len(labels_out)):
    res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
with open("prediction.txt", 'w') as f_obj:
    f_obj.write(res)