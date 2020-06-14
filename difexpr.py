import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

path = 'prediction_06_13_17_19_56.txt'
path_tr = '../found_transcripts.fasta'
df = pd.read_csv(path, header=None, sep=' ')
df.columns = ['transcript', 'probability']
"""
axes = plt.gca()
axes.set_xlim([0.0, 1.0])
sns_plot = sns.kdeplot(df['probability'], shade=True)
fig = sns_plot.get_figure()
plt.show()
"""
# print(len(df) - len(df[df['probability'] >= 0.8]) - len(df[df['probability'] <= 0.3]))
lnc = list(df[df['probability'] <= 0.3]['transcript'])
out = ''
for record in SeqIO.parse(path_tr, "fasta"):
    if record.id in lnc:
        out += '>' + str(record.id) + '\n' + str(record.seq) + '\n'
with open('../found_transcripts_lnc.fasta', 'w') as f_obj:
    f_obj.write(out)
