from Bio import SeqIO
from os.path import isfile
from time import time
from sys import stdout
import numpy as np


class OrfFinder:
    def __init__(self, seq):
        self.seq = seq
        self.fr_list = None
        self.orf = None
        self.uorf = None
        self.orf_coord1 = None
        self.uorf_coord1 = None
        self.orf_coord2 = None
        self.uorf_coord2 = None
        self.stops = None
        self.ptc = 0
        self.orfs = []
        self.orf_uorf = []
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
                       'G': ['GGT', 'GGC', 'GGA', 'GGG'], 'TTT'
                       'stop': ['TAA', 'TAG', 'TGA'],
                       'M': ['ATG'], 'W': ['TGG'],
                       }

    def find_orfs(self):
        """defines three probable frames and
           a list of all ORFs for three stop
           codons;
           finds upstream ORFs if they exist
        """
        n = 3
        seq = self.seq

        seq_len = len(seq)
        fr_1 = [seq[i:i + n] for i in range(0, seq_len, n)]
        fr_2 = [seq[i:i + n] for i in range(1, seq_len, n)]
        fr_3 = [seq[i:i + n] for i in range(2, seq_len, n)]
        self.fr_list = [fr_1, fr_2, fr_3]
        for fr in self.fr_list:
            start_atg = (self.codon_indexer(fr, 'ATG'), 'ATG')
            """
            alternative start codons
            start_ctg = (self.codon_indexer(fr, 'CTG'), 'CTG')
            start_gtg = (self.codon_indexer(fr, 'GTG'), 'GTG')
            start_ttg = (self.codon_indexer(fr, 'TTG'), 'TTG')
            start_acg = (self.codon_indexer(fr, 'ACG'), 'ACG')
            start_atc = (self.codon_indexer(fr, 'ATC'), 'ATC')
            start_att = (self.codon_indexer(fr, 'ATT'), 'ATT')
            start_aag = (self.codon_indexer(fr, 'AAG'), 'AAG')
            start_ata = (self.codon_indexer(fr, 'ATA'), 'ATA')
            start_agg = (self.codon_indexer(fr, 'AGG'), 'AGG')
            """
            stop_taa = (self.codon_indexer(fr, 'TAA'), 'TAA')
            stop_tag = (self.codon_indexer(fr, 'TAG'), 'TAG')
            stop_tga = (self.codon_indexer(fr, 'TGA'), 'TGA')
            starts = [start_atg]  # , start_ctg, start_acg, start_gtg, start_ttg,
            # start_atc, start_att, start_aag, start_ata, start_agg]
            stops = [stop_taa, stop_tag, stop_tga]
            self.stops = stops
            for start in starts:
                start_pos, start_cod = start
                for stop in stops:
                    stop_pos, stop_cod = stop
                    self.orf_check(fr, start_pos, stop_pos)

    def orf_check(self, fr, start_l, stop_l):
        """returns sequences of possible ORFs and uORFs
        """
        if start_l:
            for start in start_l:
                if stop_l:
                    for stop in stop_l:
                        ORFlen = stop - start  # ORF length in codons
                        if ORFlen > 3:
                            frame = fr[start:stop + 1]
                            fr_seq = ''.join(n for n in frame)
                            coord = self.seq.index(fr_seq)
                            self.orfs.append((frame, coord))

    def find_stack_orf_uorf(self):
        for n in range(len(self.orfs)):
            uorf = self.orfs[n]
            for m in range(len(self.orfs)):
                if n != m:
                    orf = self.orfs[m]
                    if uorf[1] + len(uorf[0]) <= orf[1] and len(uorf[0]) <= 100:
                        self.orf_uorf.append([uorf, orf])

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

    def find_max_uorf_orf(self):
        stored_uorf = [[]]
        stored_orf = [[]]
        for uorf, orf in self.orf_uorf:
            if len(uorf[0]) >= len(stored_uorf[0]) and len(orf[0]) >= len(stored_orf[0]):
                stored_uorf = uorf
                stored_orf = orf
        if len(stored_uorf[0]) != 0:
            self.uorf = ''.join(n for n in stored_uorf[0])
            self.uorf_coord1 = stored_uorf[1]
            self.uorf_coord2 = stored_uorf[1] + len(self.uorf)
            self.orf = ''.join(n for n in stored_orf[0])
            self.orf_coord1 = stored_orf[1]
            self.orf_coord2 = stored_orf[1] + len(self.orf)
        else:
            stored_orf = ['', 0]
            for orf in self.orfs:
                if len(orf[0]) > len(stored_orf[0]):
                    stored_orf = orf
            self.orf = ''.join(n for n in stored_orf[0])
            self.orf_coord1 = stored_orf[1]
            self.orf_coord2 = stored_orf[1] + len(self.orf)
        self.find_ptc(self.orf)

    def find_ptc(self, orf):
        length = len(orf) * 3
        stops = ['TAA', 'TAG', 'TGA']
        inds = {'TAA': [], 'TAG': [], 'TGA': []}
        for stop in stops:
            stored_orf = orf
            while stop in stored_orf:
                ind = orf.index(stop)
                inds[stop].append(ind)
                stored_orf = stored_orf[ind+1:]
        ptcs = []
        for key in inds.keys():
            for ind in inds[key]:
                if 50 <= length - ind * 3 <= 55:
                    ptcs.append(ind)
        if len(ptcs) != 0:
            self.ptc = min(ptcs)


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
    of.find_orfs()
    of.find_stack_orf_uorf()
    of.find_max_uorf_orf()
    if of.uorf is not None:
        line = '>' + str(record.id) + ',' + str(of.uorf_coord1) + ':' + str(of.uorf_coord2) + ',' +\
               str(of.orf_coord1) + ':' + str(of.orf_coord2) + ',' + str(of.ptc) + '\n' + str(record.seq) + '\n'
        if not isfile(out):
            with open(out, 'w') as f_obj:
                f_obj.write(line)
        else:
            with open(out, 'a') as f_obj:
                f_obj.write('\n' + line)
    else:
        line = '>' + str(record.id) + ',0:0,0:0,' + str(of.ptc) + '\n' + str(record.seq) + '\n'
        if not isfile(out):
            with open(out, 'w') as f_obj:
                f_obj.write(line)
        else:
            with open(out, 'a') as f_obj:
                f_obj.write('\n' + line)
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
