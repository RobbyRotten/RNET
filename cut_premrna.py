import pandas as pd
from Bio import SeqIO
from os.path import isfile

gtf_path = '../assemble_all_transcripts_more_then4_grouped_1000.gtf'
genome_path = '../GCF_000002425.4_Phypa_V3_genomic_100K.fna'
out_path = '../Ppatens_pre_mRNA.fasta'
gtf = pd.read_csv(gtf_path, delimiter='\t')
for record in SeqIO.parse(genome_path, "fasta"):
    chromosome = record.id
    df = gtf[gtf['chr'] == chromosome]
    groups = set(df['gr'])
    for gr in groups:
        transcripts = df[df['gr'] == gr].reset_index()
        info = '>' + transcripts['tr_id'][0] + '_' + chromosome + ','
        if all(transcripts['str.x'] == '+'):
            start = min(transcripts['value_1'])
            stop = max(transcripts['value_2'])
            premrna = record.seq[start:stop]
            coords = pd.DataFrame([transcripts['value_1'], transcripts['value_2']]).T
            coords -= start
            for n in range(len(coords)):
                info += str(coords.loc[n]['value_1']) + ':' + str(coords.loc[n]['value_2']) + ','
            info = info[:-1]
        else:
            start = max(transcripts['value_2'])
            stop = min(transcripts['value_1'])
            rna = record.seq[stop: start]
            premrna = rna.complement()
            coords = pd.DataFrame([transcripts['value_1'], transcripts['value_2']]).T
            coords = -coords + start
            for n in range(len(coords)):
                info += str(coords.loc[n]['value_2']) + ':' + str(coords.loc[n]['value_1']) + ','
            info = info[:-1]
        out = info + '\n' + str(premrna).upper() + '\n'
        if not isfile(out_path):
            with open(out_path, 'w') as f_obj:
                f_obj.write(out)
        else:
            with open(out_path, 'a') as f_obj:
                f_obj.write(out)
        print(out)
print('Success')

