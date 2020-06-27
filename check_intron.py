path = 'retention2.txt'

with open(path, 'r') as f_obj:
    lines = f_obj.readlines()
dic = {}
for line in lines:
    spl = line.split('\t')
    if spl[0] not in dic.keys():
        dic[spl[0]] = 1
    else:
        dic[spl[0]] += 1
cnt = 0
for key in dic.keys():
    if dic[key] > 1:
        print(key, dic[key])
        cnt += dic[key] - 1
print(cnt)
