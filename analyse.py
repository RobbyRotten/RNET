import pandas as pd
from Bio import SeqIO

path_lnc = '../Ppatens_lnc.fasta'
path_tab = '../transcripts_info_deseq.txt'

df = pd.read_csv(path_tab, sep='\t')
"""'id', 'chr', 'start', 'end', 'str', 'gene.n', 'class.c', 'num.assem',
       'log2FoldChange', 'pvalue', 'padj', 'de.pv', 'is.lnc', 'Kn_smg1_l1_b1',
       'Kn_smg1_l1_b2', 'Kn_smg1_l2_b1', 'Kn_smg1_l2_b2', 'WT_smg1_b1',
       'WT_smg1_b2'"""
class_u = df[df['class.c'] == 'u']['id']
class_x = df[df['class.c'] == 'x']['id']
class_s = df[df['class.c'] == 's']['id']

u_list = list(n for n in class_u)
x_list = list(n for n in class_x)
s_list = list(n for n in class_s)

num_u = 0
num_x = 0
num_s = 0
tot = 0

for record in SeqIO.parse(path_lnc, 'fasta'):
    tot += 1
    if record.id in u_list:
        num_u += 1
    elif record.id in x_list:
        num_x += 1
    elif record.id in s_list:
        num_s += 1
print('num_u = ', num_u, ' from ', len(u_list))
print('num_x = ', num_x, ' from ', len(x_list))
print('num_s = ', num_s, ' from ', len(s_list))
print('Total = ', tot)

