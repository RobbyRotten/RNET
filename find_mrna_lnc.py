import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

path = 'prediction_06_13_17_19_56.txt'
path_tr = '../assemble_all_transcripts_more_then4.fa'
df = pd.read_csv(path, header=None, sep=' ')
df.columns = ['transcript', 'probability']
"""
axes = plt.gca()
axes.set_xlim([0.0, 1.0])
sns_plot = sns.kdeplot(df['probability'], shade=True)
fig = sns_plot.get_figure()
plt.show()
exit(0)
"""
lnc = list(df[df['probability'] <= 0.3]['transcript'])
mrna = list(df[df['probability'] >= 0.8]['transcript'])
lnc_out = ''
mrna_out = ''
for record in SeqIO.parse(path_tr, "fasta"):
    if record.id in lnc:
        lnc_out += '>' + str(record.id) + '\n' + str(record.seq) + '\n'
    elif record.id in mrna:
        mrna_out += '>' + str(record.id) + '\n' + str(record.seq) + '\n'
with open('../Ppatens_mrna_all.fasta', 'w') as f_obj:
    f_obj.write(mrna_out)
with open('../Ppatens_lnc_all.fasta', 'w') as f_obj:
    f_obj.write(lnc_out)
