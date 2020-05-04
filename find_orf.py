from Bio import SeqIO
from os.path import isfile
from time import time
from sys import stdout


class OrfFinder:
    def __init__(self, seq):
        self.seq = seq
        self.fr_list = None
        self.orf = None
        self.uorf = None
        self.orfs = []
        self.ptcs = []
        self.uorf_cands = []
        self.uorfs = []
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
                       'G': ['GGT', 'GGC', 'GGA', 'GGG'],'TTT'
                       'stop': ['TAA', 'TAG', 'TGA'],
                       'M': ['ATG'], 'W': ['TGG'],
                       }

    def find_orf(self, flag):
        """defines three probable frames and
           a list of all ORFs for three stop
           codons;
           finds upstream ORFs if they exist
        """
        n = 3
        seqs = None
        if flag == 'orf':
            seqs = [self.seq]
            for seq in seqs:
                seq_len = len(seq)
                fr_1 = [seq[i:i + n] for i in range(0, seq_len, n)]
                fr_2 = [seq[i:i + n] for i in range(1, seq_len, n)]
                fr_3 = [seq[i:i + n] for i in range(2, seq_len, n)]
                self.fr_list = [fr_1, fr_2, fr_3]
                frame_num = 0
                for fr in self.fr_list:
                    frame_num += 1
                    start_atg = (self.codon_indexer(fr, 'ATG'), 'ATG')
                    # alternative start codons
                    # start_ctg = (self.codon_indexer(fr, 'CTG'), 'CTG')
                    # start_gtg = (self.codon_indexer(fr, 'GTG'), 'GTG')
                    # start_ttg = (self.codon_indexer(fr, 'TTG'), 'TTG')

                    stop_taa = (self.codon_indexer(fr, 'TAA'), 'TAA')
                    stop_tag = (self.codon_indexer(fr, 'TAG'), 'TAG')
                    stop_tga = (self.codon_indexer(fr, 'TGA'), 'TGA')
                    starts = [start_atg]  # , start_ctg, start_ttg, start_gtg]
                    stops = [stop_taa, stop_tag, stop_tga]
                    for start in starts:
                        start_pos, start_cod = start
                        for stop in stops:
                            stop_pos, stop_cod = stop
                            self.orf_check(fr, start_pos, stop_pos, flag, start_cod, stop_cod, frame_num)

        elif flag == 'uorf':
            seqs = self.uorf_cands
            for fr, fr_num, start_coord in seqs:
                start_atg = (self.codon_indexer(fr, 'ATG'), 'ATG')
                start_ctg = (self.codon_indexer(fr, 'CTG'), 'CTG')
                start_gtg = (self.codon_indexer(fr, 'GTG'), 'GTG')
                start_ttg = (self.codon_indexer(fr, 'TTG'), 'TTG')
                start_acg = (self.codon_indexer(fr, 'ACG'), 'ACG')
                start_atc = (self.codon_indexer(fr, 'ATC'), 'ATC')
                start_att = (self.codon_indexer(fr, 'ATT'), 'ATT')
                start_aag = (self.codon_indexer(fr, 'AAG'), 'AAG')
                start_ata = (self.codon_indexer(fr, 'ATA'), 'ATA')
                start_agg = (self.codon_indexer(fr, 'AGG'), 'AGG')

                stop_taa = (self.codon_indexer(fr, 'TAA'), 'TAA')
                stop_tag = (self.codon_indexer(fr, 'TAG'), 'TAG')
                stop_tga = (self.codon_indexer(fr, 'TGA'), 'TGA')
                starts = [start_atg, start_ctg, start_acg, start_gtg, start_ttg,
                          start_atc, start_att, start_aag, start_ata, start_agg]
                stops = [stop_taa, stop_tag, stop_tga]
                for start in starts:
                    start_pos, start_cod = start
                    for stop in stops:
                        stop_pos, stop_cod = stop
                        self.orf_check(fr, start_pos, stop_pos, flag, start_cod, stop_cod, fr_num)

    def orf_check(self, fr, start_l, stop_l, flag, start_cod, stop_cod, frame_num):
        """returns sequences of possible ORFs and uORFs
        """
        if start_l:
            for start in start_l:
                if stop_l:
                    for stop in stop_l:
                        ORFlen = stop - start  # ORF length in codons
                        if ORFlen > 3:
                            frame = fr[start:stop+1]
                            upframe = fr[:start]
                            if flag == 'orf':
                                self.orfs.append((frame, frame_num, start))
                                if start_cod in upframe:
                                    if stop_cod in upframe:
                                        self.uorf_cands.append((upframe, frame_num, start))
                            elif flag == 'uorf' and len(frame) != len(fr):
                                self.uorfs.append((frame, frame_num, start))

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

    def find_orf_for_uorf(self, orfs):
        out_orfs = []
        for orf in orfs:
            if self.uorf[2] + len(self.uorf[0]) <= orf[2] and self.uorf[1] == orf[1]:
                out_orfs.append(orf)
        if len(out_orfs) != 0:
            return max(out_orfs)
        else:
            return ["Empty"]

    def find_max_uorf(self, uorfs):
        stored = []
        for uorf in uorfs:
            if len(uorf[0]) > len(stored):
                stored = uorf
        return stored

    def find_best_orf(self, flag):
        """finds best the most possible ORF according to
           the effectiveness of start codons
        """
        if flag == 'orf':
            orfs_atg = []
            orfs_ctg = []
            orfs_ttg = []
            orfs_gtg = []
            for fr in self.orfs:
                if fr[0][0] == 'ATG':
                    orfs_atg.append(fr)
                elif fr[0][0] == 'CTG':
                    orfs_ctg.append(fr)
                elif fr[0][0] == 'TTG':
                    orfs_ttg.append(fr)
                elif fr[0][0] == 'GTG':
                    orfs_gtg.append(fr)
            if len(orfs_atg) != 0:
                self.orf = self.find_orf_for_uorf(orfs_atg)
            elif len(orfs_ctg) != 0:
                self.orf = self.find_orf_for_uorf(orfs_ctg)
            elif len(orfs_gtg) != 0:
                self.orf = self.find_orf_for_uorf(orfs_gtg)
            elif len(orfs_ttg) != 0:
                self.orf = self.find_orf_for_uorf(orfs_ttg)

        elif flag == 'uorf':
            orfs_atg = []
            orfs_ctg = []
            orfs_ttg = []
            orfs_gtg = []
            orfs_acg = []
            orfs_atc = []
            orfs_att = []
            orfs_aag = []
            orfs_ata = []
            orfs_agg = []
            for fr in self.uorfs:
                if fr[0][0] == 'ATG':
                    orfs_atg.append(fr)
                elif fr[0][0] == 'CTG':
                    orfs_ctg.append(fr)
                elif fr[0][0] == 'TTG':
                    orfs_ttg.append(fr)
                elif fr[0][0] == 'GTG':
                    orfs_gtg.append(fr)
                elif fr[0][0] == 'ACG':
                    orfs_acg.append(fr)
                elif fr[0][0] == 'ATC':
                    orfs_atc.append(fr)
                elif fr[0][0] == 'ATT':
                    orfs_att.append(fr)
                elif fr[0][0] == 'AAG':
                    orfs_aag.append(fr)
                elif fr[0][0] == 'ATA':
                    orfs_ata.append(fr)
                elif fr[0][0] == 'AGG':
                    orfs_agg.append(fr)
            if len(orfs_atg) != 0:
                self.uorf = self.find_max_uorf(orfs_atg)
            elif len(orfs_ctg) != 0:
                self.uorf = self.find_max_uorf(orfs_ctg)
            elif len(orfs_gtg) != 0:
                self.uorf = self.find_max_uorf(orfs_gtg)
            elif len(orfs_ttg) != 0:
                self.uorf = self.find_max_uorf(orfs_ttg)
            elif len(orfs_acg) != 0:
                self.uorf = self.find_max_uorf(orfs_acg)
            elif len(orfs_att) != 0:
                self.uorf = self.find_max_uorf(orfs_att)
            elif len(orfs_ata) != 0:
                self.uorf = self.find_max_uorf(orfs_ata)
            elif len(orfs_atc) != 0:
                self.uorf = self.find_max_uorf(orfs_atc)
            elif len(orfs_aag) != 0:
                self.uorf = self.find_max_uorf(orfs_aag)
            elif len(orfs_agg) != 0:
                self.uorf = self.find_max_uorf(orfs_agg)


seq_path = '../test_transcript.fasta'
out = 'uorfs.fasta'
with open(seq_path, 'r') as f_obj:
    content = f_obj.read()
total_num = content.count('>')
del content
cnt = 0
t0 = time()
for record in SeqIO.parse(seq_path, "fasta"):
    cnt += 1
    of = OrfFinder(str(record.seq).upper())
    of.find_orf('orf')
    of.find_orf('uorf')
    of.find_best_orf('uorf')
    of.find_best_orf('orf')

    print(of.orf)

    pr = False
    for fr, fr_num, start_coord in of.uorf_cands:
        for n in range(len(fr)):
            if fr[n] == 'ATG' and fr[n+1] == 'CAA' and pr is False and len(fr) > 8:
                print(fr)
                pr = True
    print(of.uorf)
    """
    if len(of.uorfs) != 0:
        if not isfile(out):
            with open(out, 'w') as f_obj:
                f_obj.write('>' + record.id + '_uorf\n' + of.uorf)
        else:
            with open(out, 'a') as f_obj:
                f_obj.write('\n>' + record.id + '_uorf\n' + of.uorf)
    t2 = time()
    done = int(50 * cnt / total_num)
    stdout.write("\rTotal time: {:.2f} min | [{}{}{}] {}/{} ({}%) complete ".format((t2 - t0) / 60,
                                                                                    '=' * done,
                                                                                    '>',
                                                                                    ' ' * (50 - done),
                                                                                    cnt,
                                                                                    total_num,
                                                                                    str(2 * done)))
    stdout.flush()
print('\nSuccess')
"""