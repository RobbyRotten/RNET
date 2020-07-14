from Bio import SeqIO

path = 'alignment1/mrna_lnc_all_tab.txt'
path_de = 'found_transcripts_new.fasta'

des = []
for record in SeqIO.parse(path_de, 'fasta'):
    des.append(record.id)

diff_expr_lnc = 0
diff_expr_mrna = 0
sense_antisense = 0
with open(path, 'r') as f_obj:
    lines = f_obj.readlines()
for line in lines:
    spl = line.split('\t')
    if spl[4] == '+' and spl[9] == '-':
        sense_antisense += 1
    if spl[6] in des:
        diff_expr_lnc += 1
    if spl[1] in des:
        diff_expr_mrna += 1
print(sense_antisense, diff_expr_mrna, diff_expr_lnc)
