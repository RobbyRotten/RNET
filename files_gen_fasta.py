from Bio import SeqIO
import sys
import numpy as np

filename = sys.argv[1]
records = []
for record in SeqIO.parse(filename, 'fasta'):
    records.append(record)
print('Got records')
numbers = [n for n in range(len(records))]
test_len = int(len(numbers) * 0.1)
test_num = []
n = 0
while n < test_len:
    popped_num = numbers.pop(np.random.randint(0, len(numbers)))
    if popped_num in test_num:
        while popped_num not in test_num:
            popped_num = numbers.pop(np.random.randint(0, len(numbers)))
    test_num.append(popped_num)
    n += 1
print('Got numbers')
test_records = []
train_records = []
for n in range(len(records)):
    if n in test_num:
        test_records.append(records[n])
    else:
        train_records.append(records[n])

print('Got lists')
test_out = ''
for record in test_records:
    test_out += '>' + str(record.name) + '\n' + str(record.seq) + '\n'
with open(filename[:-6] + '_test.fasta', 'w') as f_obj:
    f_obj.write(test_out)

train_out = ''
for record in train_records:
    train_out += '>' + str(record.name) + '\n' + str(record.seq) + '\n'
with open(filename[:-6] + '_train.fasta', 'w') as f_obj:
    f_obj.write(train_out)
print('Done')
print(len(records), len(test_records), len(train_records)) 
