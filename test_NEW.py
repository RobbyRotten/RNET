from net_v5 import Nnet
import pandas as pd

path_cd = "crossval/fold_1/train_cd.csv"
coding = pd.read_csv(path_cd, sep=';')
coding = coding.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0)

path_nc = "crossval/fold_1/train_nc.csv"
noncoding = pd.read_csv(path_nc, sep=';')
noncoding = noncoding.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0)

path_qr = "crossval/fold_1/test.csv"
query = pd.read_csv(path_qr, sep=';')
query_names = query.Name
query = query.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0)

nnet = Nnet(data_cd=coding, data_nc=noncoding, data_qr=query, layers_num=7, epochs=15000, threads=2)
del coding, noncoding, query
nnet.preprocessing()
# nnet.model_by_layers()
nnet.set_model()
nnet.train()
labels_out = nnet.predict()
# labels_out = [str(elem[0])[1:-1] for elem in nnet.out]
# nnet.save_model()
# nnet.load_model()
print(labels_out)

res = ''
for n in range(len(labels_out)):
    res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
with open("prediction.txt", 'w') as f_obj:
    f_obj.write(res)
