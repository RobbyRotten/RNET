import os
import shutil
import numpy as np
from Bio import SeqIO

fasta_cd = ''
fasta_nc = ''

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
