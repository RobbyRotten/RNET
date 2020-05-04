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
        self.orf_coord = None
        self.uorf_coord = None
        self.orfs = []
        self.orf_uorf = []
        self.ptcs = []
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
                            self.orfs.append((frame, start))

    def find_stack_orf_uorf(self):
        for n in range(len(self.orfs)):
            uorf = self.orfs[n]
            for m in range(len(self.orfs)):
                if n != m:
                    orf = self.orfs[m]
                    if uorf[1] + len(uorf[0]) <= orf[1]:
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
        stored_uorf = stored_uorf[0]
        stored_orf = stored_orf[0]
        self.uorf = ''.join(n for n in stored_uorf)
        self.orf = ''.join(n for n in stored_orf)
        self.uorf_coord = self.seq.index(self.uorf) + 1
        self.orf_coord = self.seq.index(self.orf) + 1


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
        line = '>' + record.id + '_uorf-coord:' + str(of.uorf_coord) + '\n' + of.uorf +\
               '\n>' + record.id + '_orf-coord:' + str(of.orf_coord) + '\n' + of.orf
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
