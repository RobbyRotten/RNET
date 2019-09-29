import sys
import numpy as np

file = sys.argv[1]
with open(file, 'r') as f_obj:
    lines = f_obj.readlines()
total = len(lines)
test_lines = []
n = 0
while n < int(total * 0.1):
    ind = np.random.randint(1, len(lines))
    test_lines.append(lines.pop(ind))
    n += 1

test_out = lines[0]
for line in test_lines:
    test_out += line
file_test = file[:-4] + '_test.csv'
with open(file_test, 'w') as f_obj:
    f_obj.write(test_out)
print(file_test)

file_train = file[:-4] + '_train.csv'    
train_out = ''
for line in lines:
    train_out += line
with open(file_train, 'w') as f_obj:
    f_obj.write(train_out)
print(file_train)
