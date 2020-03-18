from Bio.SeqUtils import GC
from Bio import SeqIO
import pandas as pd
import sys

from orf_handler_v2 import ORFHandler


class Parser:
    """Deals with the whole fasta file.
    """
    def __init__(self, filename):
        self.filename = filename
        self.names = []
        self.GCs = []
        self.TRlens = []
        self.ORFlens = []
        self.ORFcovs = []
        self.ORFcovs_mean = []
        self.insts = []
        self.fscores = []
        self.opt_cod_frs = []
        self.entr_sums = []
        self.RCBs = []
        self.RSCUs = []  # RSCU (Relative Synonymous Codon Usage)
        self.SCUOs = []  # SCUO (Synonymous Codon Usage Order)
        self.hex_freq = []

    def parse(self):
        print('-Estimating file volume...')
        with open(self.filename) as f:
            num = f.read().count('>')
        cnt = 0
        print('-Extracting features...')
        for record in SeqIO.parse(self.filename, "fasta"):
            cnt += 1
            done = int(50 * cnt / num)
            sys.stdout.write("\r[%s%s] %s" % ('=' * done, ' ' * (50 - done),
                             str(cnt) + '/' + str(num) + ' ' + str(2 * done) + '% complete'))
            sys.stdout.flush()

            self.names.append(record.id)
            self.GCs.append(GC(record.seq))
            orf = ORFHandler(record.seq)
            orf.ORF_former()
            orf.find_max_orf()
            orf.pept_former()
            orf.pept_counter()
            orf.syn_codons()
            orf.fickett_count()
            orf.rcb_counter()
            orf.entropy_counter()
            orf.mean_orf_counter()
            orf.hexamer_cnt()
            self.ORFlens.append(orf.orf_len)
            self.opt_cod_frs.append(orf.opt_cod_fr)
            self.RSCUs.append(orf.RSCU)
            self.TRlens.append(orf.length)
            self.ORFcovs.append(orf.orf_cov)
            self.insts.append(orf.inst)
            self.fscores.append(orf.fscore)
            self.RCBs.append(orf.RCB)
            self.entr_sums.append(orf.entr_sum)
            self.ORFcovs_mean.append(orf.mean_orf_cov)
            self.SCUOs.append(orf.SCUO)
            self.hex_freq.append(orf.hex_dic)
        print('\n-Features extraction complete')

    def count_hex(self):
        """returns a list containing
           hexamers occured in a sequence
           for all sequences in fasta file.
        """
        hex_all = []
        for record in SeqIO.parse(self.filename, "fasta"):
            seq = str(record.seq)
            hex_list = []
            for hexamer in self.hex_freq[0].keys():
                cnt = seq.count(hexamer)
                if cnt != 0:
                    hex_list.append(hexamer)
            hex_all.append(hex_list)
        return hex_all

    def gen_hex_tab(self):
        """returns a series object with
           sum of hexamer frequencies for
           all fasta file.
        """
        df = pd.DataFrame(self.hex_freq)
        freqs = df.sum(axis=0)
        df_fr = pd.DataFrame(freqs)
        df_fr.to_csv('%s_hex_freq.csv' % self.filename[:-6], sep=";")
        return freqs

    def gen_feat_tab(self):
        s_names = pd.Series(self.names)
        s_GCs = pd.Series(self.GCs)
        s_TRlens = pd.Series(self.TRlens)
        s_ORFlens = pd.Series(self.ORFlens)
        s_ORFcovs = pd.Series(self.ORFcovs)
        s_ORFcovs_mean = pd.Series(self.ORFcovs_mean)
        s_insts = pd.Series(self.insts)
        s_fscores = pd.Series(self.fscores)
        s_codfrs = pd.Series(self.opt_cod_frs)
        s_rcbs = pd.Series(self.RCBs)
        s_entrs = pd.Series(self.entr_sums)
        s_scuos = pd.Series(self.SCUOs)
        df = pd.DataFrame([s_names,
                           s_GCs,
                           s_TRlens,
                           s_ORFlens,
                           s_ORFcovs,
                           s_insts,
                           s_fscores,
                           s_codfrs,
                           s_rcbs,
                           s_entrs,
                           s_ORFcovs_mean,
                           s_scuos,
                           ]).T
        df.columns = ['Name',
                      'GC',
                      'Tr_len',
                      'ORF_len',
                      'ORF_cov',
                      'Inst_ind',
                      'Fcktt_score',
                      'Opt_cod_frq',
                      'RCB',
                      'RelEntSum',
                      'mnORF_cov',
                      'SCUO',
                      ]
        return df

    def rscu_tab(self, save):
        df = pd.DataFrame(self.RSCUs)
        df.index = self.names
        if save:
            df.to_csv('%s_rscu.csv' % self.filename[:-6], sep=";")
        return df
