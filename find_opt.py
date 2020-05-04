from os import listdir
from sys import argv


path = argv[1]
files = listdir(path)
max_file_auc = ''
max_file_acc = ''
max_auc = '0'
max_acc = '0'
for f in files:
    if '.png' in f:
        name = f.split('_')
        auc = name[5]
        acc = name[6]
        if auc > max_auc:
            max_file_auc = f
            max_auc = auc
        if acc > max_acc:
            max_file_acc = f
            max_acc = acc
print('Max AUC: ', max_file_auc)
print('Max acc: ', max_file_acc)
