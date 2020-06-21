path = '../aligned_all.tab'
path_tab = 'aligned_tab.txt'
D = 10

with open(path, 'r') as f_obj:
    lines = f_obj.readlines()
out = ''  # 'score	name1	start1	alnSize1	strand1	seqSize1	name2	start2	alnSize2	strand2	seqSize2	blocks\n'
for line in lines:
    if 'TCONS' in line:
        out += line
with open('aligned_all_tab.txt', 'w') as f_obj:
    f_obj.write(out)
