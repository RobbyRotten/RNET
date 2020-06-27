from Bio import SeqIO
import pandas as pd

path_pred = 'prediction_06_13_17_19_56.txt'
# path_lnc = '../found_transcripts_lnc.fasta'
path = '../assemble_all_transcripts_more_then4.fa'
fout = '../Ppatens_mrna.fasta'

lncs = []
df = pd.read_csv(path, header=None, sep=' ')
df.columns = ['transcript', 'probability']
lnc = list(df[df['probability'] <= 0.3]['transcript'])
out = ''
for record in SeqIO.parse(path, 'fasta'):
    if record.id not in lncs:
        out += '>' + record.id + '\n' + str(record.seq) + '\n'
with open(fout, 'w') as f_obj:
    f_obj.write(out)
