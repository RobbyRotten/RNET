from DSFormer import DSFormer
from NNet import NNet
import pandas as pd
from torch.utils.data import DataLoader

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

D = DSFormer(coding, noncoding, query)
D.preprocessing()
train = D.get_train()
test = D.get_query()
N = NNet(train, test, 2)
N.train_model()
prediction = N.predict()
res = ''
for n in range(len(prediction)):
    res += query_names[n] + ' ' + prediction[n] + '\n'
with open("prediction_trch.txt", 'w') as f_obj:
    f_obj.write(res)
