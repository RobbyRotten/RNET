import pandas as pd

gtf_path = '../assemble_all_transcripts_more_then4.gtf'
gtf_path_new = '../assemble_all_transcripts_more_then4_grouped.gtf'

gtf = pd.read_csv(gtf_path, delimiter='\t')
tr_gene_id = gtf['id.buf'].str.split(';', expand=True)
tr_gene_id.columns = ['gene_id', 'tr_id', 'null']
gene_id = tr_gene_id['gene_id'].str[9:-1]
tr_id = tr_gene_id['tr_id'].str[16:-1]
gtf = gtf.drop(['id.buf'], axis=1)
gtf = pd.concat([gtf, tr_id, gene_id], axis=1)
# gtf.to_csv(gtf_path_new)

# gtf = pd.read_csv(gtf_path_new).drop(['Unnamed: 0'], axis=1)
gene_ids = set(gtf['gene_id'])
groups = []
gr = 0
err = ''
for gene_id in gene_ids:
    df = gtf[gtf['gene_id'] == gene_id].reset_index(drop=True)
    gr_df = pd.DataFrame({'gr': list(gr for n in range(len(df)))})
    if all(df['str.x'] == '+'):
        df = df.sort_values(by='value_1')
        df = df.join(gr_df)
        groups.append(df)
        gr += 1
    elif all(df['str.x'] == '-'):
        df = df.sort_values(by='value_2', ascending=False)
        df = df.join(gr_df)
        groups.append(df)
        gr += 1
    else:
        err += 'error at gene_id = ' + df['gene_id'][0] + '\n'
with open('errors.txt', 'w') as f_obj:
    f_obj.write(err)
grouped = pd.concat(groups, axis=0).reset_index(drop=True)
chromosome = pd.DataFrame({'chr': grouped['coord.n_1']})
grouped = grouped.join(chromosome)
grouped.to_csv(gtf_path_new, sep='\t', index=False)
