import pandas as pd
from Bio import SeqIO

# tr_path = '/data7a/bio/moss_nmd/data/input/big/transcripts_info_deseq.txt'
# seq_path = '/data7a/bio/moss_nmd/data/input/big/Cuffcom_V2/assemble_all_transcripts_more_then4.fa'
tr_path = '../transcripts_info_deseq.txt'
seq_path = '../assemble_all_transcripts_more_then4.fa'
transcripts = pd.read_csv(tr_path, delimiter='\t')
expressed = transcripts[transcripts['de.pv'] > 0]

ids = list(transcripts['id'])
out = ''
for record in SeqIO.parse(seq_path, "fasta"):
    if record.id in ids and len(record.seq) >= 8:
        out += '>' + str(record.id) + '\n' + str(record.seq) + '\n'
with open('found_transcripts.fasta', 'w') as f_obj:
    f_obj.write(out)
