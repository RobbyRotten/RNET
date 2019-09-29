import numpy as np
import math
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class ORFHandler:
    """ Deals with a single sequence from fasta file.
    """
    def __init__(self, seq):
        # inclass attributes
        self.fr_list = None
        self.pept = None
        self.orfs = []
        self.seq = str(seq)
        self.max_orf = None
        self.acid_us = {}
        self.rscu_acid = {}    # rscu grouped by acid

        # output attributes
        self.length = len(self.seq)
        self.orf_len = 0       # longest orf length
        self.inst = 0          # predicted peptide instability index
        self.opt_cod_fr = 0    # optimal codon frequency
        self.RSCU = 0          # relative synonymous codon usage
        self.orf_cov = 0       # orf coverage
        self.fscore = 0        # fickett score
        self.RCB = 0           # relative codon bias
        self.entr_sum = 0      # sum of relative entropy
        self.SCUO = 0          # synonymous codon usage order
        self.mean_orf_cov = 0  # overall orf coverage
        self.hex_dic = {}

        self.codons = {'F': ['TTT', 'TTC'],
                       'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
                       'I': ['ATT', 'ATC', 'ATA'],
                       'V': ['GTT', 'GTC', 'GTA', 'GTG'],
                       'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
                       'P': ['CCT', 'CCC', 'CCA', 'CCG'],
                       'T': ['ACT', 'ACC', 'ACA', 'ACG'],
                       'A': ['GCT', 'GCC', 'GCA', 'GCG'],
                       'Y': ['TAT', 'TAC'],
                       'H': ['CAT', 'CAC'],
                       'Q': ['CAA', 'CAG'],
                       'N': ['AAT', 'AAC'],
                       'K': ['AAA', 'AAG'],
                       'D': ['GAT', 'GAC'],
                       'E': ['GAA', 'GAG'],
                       'C': ['TGT', 'TGC'],
                       'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
                       'G': ['GGT', 'GGC', 'GGA', 'GGG'],
                       'stop': ['TAA', 'TAG', 'TGA'],
                       'M': ['ATG'], 'W': ['TGG'],
                       }

    def ORF_former(self):
        """defines three probable frames and
           a list of all ORFs for three stop
           codons.
        """
        n = 3
        seq_len = len(self.seq)
        fr_1 = [self.seq[i:i+n] for i in range(0, seq_len, n)]
        fr_2 = [self.seq[i:i+n] for i in range(1, seq_len, n)]
        fr_3 = [self.seq[i:i+n] for i in range(2, seq_len, n)]
        self.fr_list = [fr_1, fr_2, fr_3]
        for fr in self.fr_list:
            start_l = self.codon_indexer(fr, 'ATG')
            stop_taa = self.codon_indexer(fr, 'TAA')
            stop_tag = self.codon_indexer(fr, 'TAG')
            stop_tga = self.codon_indexer(fr, 'TGA')
            self.ORF_check(fr, start_l, stop_taa)
            self.ORF_check(fr, start_l, stop_tag)
            self.ORF_check(fr, start_l, stop_tga)

    def hexamer_cnt(self):
        """forms a hexamer dictionary
           and couns every hexamer
           occyrency in the sequence.
        """
        hex_list = []
        bases = ['A', 'T', 'G', 'C']
        for i in bases:
            for j in bases:
                for k in bases:
                    for l in bases:
                        for m in bases:
                            for n in bases:
                                hexamer = ''.join([i, j, k, l, m, n])
                                hex_list.append(hexamer)
        for hexamer in hex_list:
            self.hex_dic[hexamer] = self.seq.count(hexamer)

    def entropy_counter(self):
        """counts weighted sum of
           relative entropy.
        """
        rel_entr_list = []
        norm_entr_list = []
        for acid, rscus in self.rscu_acid.items():
            if len(rscus) > 0:
                weight = self.acid_us[acid]
                rs = []
                for rscu in rscus:
                    if rscu > 0:
                        rs.append(rscu * math.log2(rscu))
                    else:
                        rs.append(0)
                #entr = - weight * sum([rscu * math.log2(rscu) for rscu in rscus])
                entr = - weight * sum(rs)
                cod_deg = math.log2(len(rscus))
                rel_entr = entr / cod_deg
                norm_entr = (cod_deg - entr) / cod_deg
                rel_entr_list.append(rel_entr)
                norm_entr_list.append(norm_entr)
        self.entr_sum = sum(rel_entr_list)
        self.SCUO = sum(norm_entr_list)

    def find_max_orf(self):
        """finds the longest ORF and
           counts its length and coverage.
        """
        lens = [len(orf) for orf in self.orfs]
        arr = np.array(lens)
        if len(arr) > 0:
            ind = int(np.argmax(arr))
            self.max_orf = self.orfs[ind]
            self.orf_len = len(self.max_orf) * 3
            self.orf_cov = self.orf_len / self.length
        else:
            self.orf_len = 0
            self.orf_cov = 0

    def rcb_counter(self):
        """counts relative codon bias.
        """
        cod_fr = {}
        nums = []
        if self.max_orf:
            for codon in self.max_orf:
                if codon not in cod_fr.keys():
                    cod_fr[codon] = self.max_orf.count(codon)
            base_dic = self.base_counter()
            for codon in cod_fr.keys():
                cod_en = {num: base for num, base in enumerate(codon)}
                cod_values = []
                for base in base_dic.keys():
                    spl = base.split('_')
                    if spl[0] == cod_en[int(spl[1])]:
                        cod_values.append(base_dic[base])
                cod_values = np.array(cod_values)
                multi = np.multiply.reduce(cod_values)
                num = ((cod_fr[codon] - multi) / multi) + 1
                nums.append(math.log(num))
            res = sum(nums) / len(self.max_orf)
            self.RCB = math.exp(res)
        else:
            self.RCB = 0

    def base_counter(self):
        """counts base number at 1st, 2nd
           and 3rd positions in codons.
        """
        base_dic = {}
        for base in ['A', 'T', 'G', 'C']:
            for pos in range(3):
                key = '_'.join([base, str(pos)])
                cnt = 0
                if key not in base_dic.keys():
                    for codon in self.max_orf:
                        if codon[pos] == base:
                            cnt += 1
                    base_dic[key] = cnt
        return base_dic

    def pept_former(self):
        """forms a peptide sequence
           from max ORF.
        """
        pept = ''
        if self.max_orf:
            for codon in self.max_orf:
                for acid, triples in self.codons.items():
                    if codon in triples:
                        pept += acid
            self.pept = pept

    def codon_indexer(self, fr, codon):
        """returns indexes of a codon
           in a frame cause index() func can`t be used.
        """
        if codon in fr:
            cod_list = []
            dic = {ind: cod for ind, cod in enumerate(fr)}
            for ind, cod in dic.items():
                if cod == codon:
                    cod_list.append(ind)
            return cod_list
        else:
            return None

    def ORF_check(self, fr, start_l, stop_l):
        """returns sequences of possible ORFs.
        """
        if start_l:
            for start in start_l:  # can be used with alternative starts
                if stop_l:
                    for stop in stop_l:
                        ORFlen = stop - start  # ORF length in codons
                        if ORFlen > 3:
                            frame = fr[start:stop]
                            if 'TAA' not in frame and 'TAG' not in frame and 'TGA' not in frame:
                                self.orfs.append(frame)

    def mean_orf_counter(self):
        """counts orf lengths and overall
           orf coverage.
        """
        orf_lens = np.array([len(orf) for orf in self.orfs])
        self.mean_orf_cov = np.mean(orf_lens) / self.length

    def syn_codons(self):
        """counts relative synonymous codon usage
           as average for three probable frames and
           frequency of optimal codons.
        """
        acid_us = {}
        usage = {}
        cod_freq = {}
        fr_len = 0
        opt_triples = 0
        for fr in self.fr_list:
            fr_len += len(fr)
            for acid, triples in self.codons.items():
                if acid != 'M' and acid != 'W' and acid != 'stop':
                    for triple in triples:
                        num = fr.count(triple)
                        if triple in usage.keys():
                            usage[triple] += num
                        else:
                            usage[triple] = num
        # fr_len /= 3
        # for triple in usage.keys():
        #     usage[triple] /= 3
        fr_len_div = fr_len / 3
        for acid, triples in self.codons.items():
            self.rscu_acid[acid] = []
            acid_us[acid] = []
            if acid != 'M' and acid != 'W' and acid != 'stop':
                cod_nums = list(int(usage[triple]) for triple in triples)  # ; print(cod_nums, triples)
                # opt_triple = triples[int(np.argmax(np.array(cod_nums)))]
                opt_triples += max(cod_nums)
                cod_sum = sum(cod_nums)
                if cod_sum != 0:
                    for triple in triples:
                        acid_us[acid].append(usage[triple] / 3)  # average usage
                        freq = usage[triple] / cod_sum  # overall usage divided by overall sum
                        freq = float('{:.5}'.format(freq))
                        cod_freq[triple] = freq
                        self.rscu_acid[acid].append(freq)
            self.acid_us[acid] = sum(acid_us[acid]) / fr_len_div
        opt_cod_fr = opt_triples / fr_len
        self.opt_cod_fr = float('{:.6}'.format(opt_cod_fr))
        self.RSCU = cod_freq

    def fickett_count(self):
        """counts fickett score.
        """
        ind_1 = [n for n in range(0, len(self.seq), 3)]
        ind_2 = [n for n in range(1, len(self.seq), 3)]
        ind_3 = [n for n in range(2, len(self.seq), 3)]
        pos_1 = [self.seq[n] for n in ind_1]
        pos_2 = [self.seq[n] for n in ind_2]
        pos_3 = [self.seq[n] for n in ind_3]
        pos_l = [pos_1, pos_2, pos_3]
        # count of a base at positions
        a_cnt = self.pos_counter(pos_l, 'A')
        t_cnt = self.pos_counter(pos_l, 'T')
        g_cnt = self.pos_counter(pos_l, 'G')
        c_cnt = self.pos_counter(pos_l, 'C')
        cnts_conv = [self.cnt_converter('A', a_cnt)*0.26,
                     self.cnt_converter('T', t_cnt)*0.18,
                     self.cnt_converter('G', g_cnt)*0.31,
                     self.cnt_converter('C', c_cnt)*0.33,
                     ]
        # base percentage
        a_per = self.seq.count('A') / self.length
        t_per = self.seq.count('T') / self.length
        g_per = self.seq.count('G') / self.length
        c_per = self.seq.count('C') / self.length
        pers_conv = [self.per_converter('A', a_per)*0.11,
                     self.per_converter('T', t_per)*0.12,
                     self.per_converter('G', g_per)*0.15,
                     self.per_converter('C', c_per)*0.14,
                     ]
        self.fscore = sum([cnts_conv[n]*pers_conv[n] for n in range(4)])
        self.fscore = float('{:.6}'.format(self.fscore))

    def cnt_converter(self, base, cnt):
        """converts a base number at
           positions into probability value.
        """
        if cnt <= 1.1:
            if base == 'A':
                return 0.22
            elif base == 'C':
                return 0.23
            elif base == 'G':
                return 0.08
            elif base == 'T':
                return 0.09
        elif 1.1 < cnt <= 1.2:
            if base == 'A':
                return 0.20
            elif base == 'C':
                return 0.30
            elif base == 'G':
                return 0.08
            elif base == 'T':
                return 0.09
        elif 1.2 < cnt <= 1.3:
            if base == 'A':
                return 0.34
            elif base == 'C':
                return 0.33
            elif base == 'G':
                return 0.16
            elif base == 'T':
                return 0.20
        elif 1.3 < cnt <= 1.4:
            if base == 'A':
                return 0.45
            elif base == 'C':
                return 0.51
            elif base == 'G':
                return 0.27
            elif base == 'T':
                return 0.54
        elif 1.4 < cnt <= 1.5:
            if base == 'A':
                return 0.68
            elif base == 'C':
                return 0.48
            elif base == 'G':
                return 0.48
            elif base == 'T':
                return 0.44
        elif 1.5 < cnt <= 1.6:
            if base == 'A':
                return 0.58
            elif base == 'C':
                return 0.66
            elif base == 'G':
                return 0.53
            elif base == 'T':
                return 0.69
        elif 1.6 < cnt <= 1.7:
            if base == 'A':
                return 0.93
            elif base == 'C':
                return 0.81
            elif base == 'G':
                return 0.64
            elif base == 'T':
                return 0.68
        elif 1.7 < cnt <= 1.8:
            if base == 'A':
                return 0.84
            elif base == 'C':
                return 0.70
            elif base == 'G':
                return 0.74
            elif base == 'T':
                return 0.91
        elif 1.8 < cnt <= 1.9:
            if base == 'A':
                return 0.68
            elif base == 'C':
                return 0.70
            elif base == 'G':
                return 0.88
            elif base == 'T':
                return 0.97
        else:
            if base == 'A':
                return 0.94
            elif base == 'C':
                return 0.80
            elif base == 'G':
                return 0.90
            elif base == 'T':
                return 0.97

    def per_converter(self, base, per):
        """converts a base percentage
           into probability value.
        """
        if per <= 0.17:
            if base == 'A':
                return 0.21
            elif base == 'C':
                return 0.31
            elif base == 'G':
                return 0.29
            elif base == 'T':
                return 0.58
        elif 0.17 < per <= 0.19:
            if base == 'A':
                return 0.81
            elif base == 'C':
                return 0.39
            elif base == 'G':
                return 0.33
            elif base == 'T':
                return 0.51
        elif 0.19 < per <= 0.21:
            if base == 'A':
                return 0.65
            elif base == 'C':
                return 0.44
            elif base == 'G':
                return 0.41
            elif base == 'T':
                return 0.69
        elif 0.21 < per <= 0.23:
            if base == 'A':
                return 0.67
            elif base == 'C':
                return 0.43
            elif base == 'G':
                return 0.41
            elif base == 'T':
                return 0.56
        elif 0.23 < per <= 0.25:
            if base == 'A':
                return 0.49
            elif base == 'C':
                return 0.59
            elif base == 'G':
                return 0.73
            elif base == 'T':
                return 0.75
        elif 0.25 < per <= 0.27:
            if base == 'A':
                return 0.62
            elif base == 'C':
                return 0.59
            elif base == 'G':
                return 0.64
            elif base == 'T':
                return 0.55
        elif 0.27 < per <= 0.29:
            if base == 'A':
                return 0.55
            elif base == 'C':
                return 0.64
            elif base == 'G':
                return 0.64
            elif base == 'T':
                return 0.40
        elif 0.29 < per <= 0.31:
            if base == 'A':
                return 0.44
            elif base == 'C':
                return 0.51
            elif base == 'G':
                return 0.47
            elif base == 'T':
                return 0.39
        elif 0.31 < per <= 0.33:
            if base == 'A':
                return 0.49
            elif base == 'C':
                return 0.64
            elif base == 'G':
                return 0.54
            elif base == 'T':
                return 0.24
        else:
            if base == 'A':
                return 0.28
            elif base == 'C':
                return 0.82
            elif base == 'G':
                return 0.40
            elif base == 'T':
                return 0.28

    def pos_counter(self, pos_l, base):
        """counts the ratio between max and
           min of base at positions for
           fickett score count.
        """
        base_1 = pos_l[0].count(base)
        base_2 = pos_l[1].count(base)
        base_3 = pos_l[2].count(base)
        bl = [base_1, base_2, base_3]
        b_pos = max(bl) / (min(bl) + 1)
        return b_pos

    def pept_counter(self):
        """this class can be used to get some
           other peptide properties.
        """
        if self.pept:
            pa = ProteinAnalysis(self.pept)
            inst = pa.instability_index()
            if inst > 0:
                self.inst = inst
        else:
            self.inst = 100
