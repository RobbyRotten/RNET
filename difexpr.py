import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

path = 'prediction_06_13_17_19_56.txt'
path_tr = 'found_transcripts_new.fasta'
df = pd.read_csv(path, header=None, sep=' ')
df.columns = ['transcript', 'probability']
transcripts = []
for record in SeqIO.parse(path_tr, "fasta"):
    transcripts.append(record.id)
tab = []
for n in range(len(df)):
    if df.iloc[n, 0] in transcripts:
        tab.append(df.iloc[n, :])
df_new = pd.DataFrame(tab)
"""
axes = plt.gca()
axes.set_xlim([0.0, 1.0])
sns_plot = sns.kdeplot(df_new['probability'], shade=True)
fig = sns_plot.get_figure()
plt.show()
"""
print(len(df_new))
print(len(df_new[df_new['probability'] >= 0.6]))
print(len(df_new[df_new['probability'] <= 0.4]))
# print(len(df[0.3 < df['probability'] < 0.8]))

lnc = list(df_new[df_new['probability'] <= 0.4]['transcript'])
out = ''
for record in SeqIO.parse(path_tr, "fasta"):
    if record.id in lnc:
        out += '>' + str(record.id) + '\n' + str(record.seq) + '\n'
with open('found_transcripts_lnc_de_new.fasta', 'w') as f_obj:
    f_obj.write(out)

