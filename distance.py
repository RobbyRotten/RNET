from os import mkdir, listdir
from Bio import SeqIO
from sys import stdout

lncs = []
path_lnc = 'found_transcripts_new.fasta'
for record in SeqIO.parse(path_lnc, 'fasta'):
    lncs.append(record.id)
file = 'out_dist.txt'
names5 = []
names3 = []
longer5 = 0
longer3 = 0
names = []
with open(file, 'r') as f_obj:
    lines = f_obj.readlines()
for line in lines:
    spl = line.split('\t')  # ['TCONS_003385', 'shorter_5UTR (247/410.65625)\n']
    name = spl[0]
    if name in lncs:
        if '5UTR' in spl[1] and name not in names5:
            longer5 += 1
            names5.append(name)
        if '3UTR' in spl[1] and name not in names3:
            longer3 += 1
            names3.append(name)
cross = 0
for n in names5:
    if n in names3:
        cross += 1
print(len(names3), len(names5), cross)
exit(0)

folder = 'genes1/'
path = 'all_transcripts_uorfs_ptc.fasta'
path_lnc = 'found_transcripts_new.fasta'
lncs = []
record_ids = []
record_5 = []
record_3 = []
out = ''

for record in SeqIO.parse(path_lnc, 'fasta'):
    lncs.append(record.id)
for record in SeqIO.parse(path, 'fasta'):
    spl = record.id.split(',')
    length = len(record.seq)
    record_ids.append(spl[0])
    start = int(spl[2].split(':')[0])
    end = int(spl[2].split(':')[1])
    record_5.append(start)
    record_3.append(length - end)
files = listdir(folder)
cnt = 0
num = len(files)
for file in files:
    cnt += 1
    stdout.write('\r{}/{} ({:.2f}%) complete'.format(cnt, num, cnt / num * 100))
    with open(folder + file, 'r') as f_obj:
        lines = f_obj.readlines()
    if len(lines) > 1:
        vals_5 = []
        vals_3 = []
        transcripts = []
        transcript = None
        for line in lines:
            spl = line.split('\t')
            transcript = spl[6]
            if transcript in record_ids:
                transcripts.append(transcript)
                vals_5.append(record_5[record_ids.index(transcript)])
                vals_3.append(record_3[record_ids.index(transcript)])
        for transcript in transcripts:
            if transcript in lncs:
                mean_5 = sum(vals_5) / len(vals_5)
                mean_3 = sum(vals_3) / len(vals_3)
                ind = transcripts.index(transcript)
                if vals_5[ind] < mean_5:
                    out += transcript + '\t' + 'shorter_5UTR ({}/{})'.format(vals_5[ind], mean_5) + '\n'
                if vals_3[ind] < mean_3:
                    out += transcript + '\t' + 'shorter_3UTR ({}/{})'.format(vals_3[ind], mean_3) + '\n'
with open('out_dist.txt', 'w') as f_obj:
    f_obj.write(out)
