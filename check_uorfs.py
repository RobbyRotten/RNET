from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns

path = 'uorfs_ptc_cod.fasta'
path_lnc = 'found_transcripts_lnc_de_new.fasta'

all = 0
uorfs = 0
ptcs = 0
uorf_ptc = 0
lncs = []
for record in SeqIO.parse(path_lnc, 'fasta'):
    lncs.append(record.id)
print(len(lncs))

atc = 0
aac = 0
ggt = 0
aac_atc = 0
atc_ggt = 0
aac_ggt = 0
aac_atc_ggt = 0
zero = 0
total = 0
atcs, aacs, ggts = [], [], []

for record in SeqIO.parse(path, 'fasta'):
    spl = record.id.split(',')  # TCONS_000060,85:178,182:1628,0,ATC:0.00000000,AAC:0.00000000,GGT:0.00061312
    name = spl[0]
    if name in lncs:
        total += 1
        all += 1
        uorf = spl[1]
        ptc = spl[3]
        if uorf != '0:0' and ptc != '0':
            uorf_ptc += 1
            uorfs += 1
            ptcs += 1
        else:
            if uorf != '0:0':
                uorfs += 1
            if ptc != '0':
                ptcs += 1
        if len(spl) > 4:
            length = int(uorf.split(':')[1]) - int(uorf.split(':')[0])
            atc_fr = float(spl[4].split(':')[1]) / length
            aac_fr = float(spl[5].split(':')[1]) / length
            ggt_fr = float(spl[6].split(':')[1]) / length

            if atc_fr != 0:
                atcs.append(atc_fr)
            if aac_fr != 0:
                aacs.append(aac_fr)
            if ggt_fr != 0:
                ggts.append(ggt_fr)

            if atc_fr > 0 and aac_fr > 0 and ggt_fr > 0:
                aac_atc_ggt += 1
                aac += 1
                atc += 1
                ggt += 1
            elif atc_fr > 0 and aac_fr > 0:
                aac_atc += 1
                aac += 1
                atc += 1
            elif aac_fr > 0 and ggt_fr > 0:
                aac_ggt += 1
                aac += 1
                ggt += 1
            elif atc_fr > 0 and ggt_fr > 0:
                atc_ggt += 1
                atc += 1
                ggt += 1
            else:
                if atc_fr > 0:
                    atc += 1
                elif aac_fr > 0:
                    aac += 1
                elif ggt_fr > 0:
                    ggt += 1
                else:
                    zero += 1

axes = plt.gca()
# axes.set_xlim([0.0, 1.0])
sns_plot = sns.kdeplot(aacs, shade=True)
fig = sns_plot.get_figure()
sns_plot = sns.kdeplot(atcs, shade=True)
fig = sns_plot.get_figure()
sns_plot = sns.kdeplot(ggts, shade=True)
fig = sns_plot.get_figure()
plt.legend(labels=['AAC', 'ATC', 'GGT'])
plt.show()

print(aacs)
print(atcs)
print(ggts)

print('aac_atc_ggt', aac_atc_ggt)
print('aac_atc', aac_atc)
print('aac_ggt', aac_ggt)
print('atc_ggt', atc_ggt)
print('atc', atc)
print('aac', aac)
print('ggt', ggt)
print('zero', zero)
print('total', total)

# print(uorfs, ptcs, all, uorf_ptc)
