from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns

atcs = []
aacs = []
ggts = []
path = 'all_transcripts_uorfs_ptc.fasta'
for record in SeqIO.parse(path, 'fasta'):
    spl = record.id.split(',')
    uorf = spl[1]
    if uorf != '0:0' and len(spl) > 4:
        length = int(uorf.split(':')[1]) - int(uorf.split(':')[0])
        atc = float(spl[4].split(':')[1]) / length
        atcs.append(atc)
        aac = float(spl[5].split(':')[1]) / length
        aacs.append(aac)
        ggt = float(spl[6].split(':')[1]) / length
        ggts.append(ggt)

axes = plt.gca()
axes.set_xlim([0.0, 0.035])
# sns_plot = sns.kdeplot(aacs, shade=True, bw=0.1)
# fig = sns_plot.get_figure()
plt.hist(aacs, bins=200, color='red', histtype='step')
plt.hist(atcs, bins=200, color='blue', histtype='step')
plt.hist(ggts, bins=200, color='green', histtype='step')
plt.legend(labels=['AAC', 'ATC', 'GGT'])
# plt.show()

mean_atc = sum(atcs) / len(atcs)
mean_aac = sum(aacs) / len(aacs)
mean_ggt = sum(ggts) / len(ggts)
print(mean_atc * 1000, mean_aac * 1000, mean_ggt * 1000)