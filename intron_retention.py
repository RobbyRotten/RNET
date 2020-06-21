from os.path import isfile, isdir
from sys import stdout
from os import mkdir, listdir
from Bio import SeqIO
import numpy as np

path_tab = 'aligned_all_tab.txt'
folder = 'genes/'
lnc_path = '../found_transcripts_lnc.fasta'
D = 10


def find_retention():
    lncs = []
    for record in SeqIO.parse(lnc_path, 'fasta'):
        lncs.append(record.id)
    files = listdir(folder)
    fout = 'retention1.txt'
    num = len(files)
    cnt = 0
    for f in files:
        # print(f)
        out = ''
        stdout.write('\r{}/{} ({:.2f}%) complete'.format(cnt, num, cnt / num * 100))
        cnt += 1
        name = f[:-4]
        with open(folder + f, 'r') as f_obj:
            lines = f_obj.readlines()
        if len(lines) > 1:
            coords_mrna = []
            coords = []
            coords_lnc = {}
            for line in lines:
                spl = line.split('\t')
                transcript = spl[6]
                start = int(spl[7])
                end = start + int(spl[8]) - 1
                # key = '{}:{}'.format(start, end)
                if transcript in lncs:
                    if transcript not in coords_lnc.keys():
                        coords_lnc[transcript] = [start, end]
                    else:
                        coords_lnc[transcript].append(start)
                        coords_lnc[transcript].append(end)
                else:
                    coords_mrna.append([start, end])
                    coords.append(start)
                    coords.append(end)
            if len(coords_lnc) != 0 and len(coords) != 0:
                for key in coords_lnc.keys():
                    coords_lnc[key].sort()
                exons = []
                arr = np.zeros((len(coords_mrna), max(coords) + 1))
                for n in range(len(coords_mrna)):
                    arr[n][coords_mrna[n][0]:coords_mrna[n][1]] = 1
                arr_sum = np.sum(arr, axis=0).astype(int)
                on = False
                for n in range(len(arr_sum)):
                    # print(arr_sum[n], n)
                    # print(n)
                    if arr_sum[n] != 0 and not on:
                        # print('start')
                        exons.append(n)
                        on = True
                    elif arr_sum[n] == 0 and on:
                        # print('stop')
                        exons.append(n)
                        on = False
                    elif n == len(arr_sum) and on:
                        # print('stop')
                        exons.append(n)
                if exons[0] != 0:
                    introns = []
                    le = len(exons)
                    la = len(arr_sum)
                    for n in range(le):
                        # print(n, len(exons), exons[n], len(arr_sum))
                        if n == 0 and exons[n] != 0:
                            introns.append(0)
                        if n != le - 1:
                            if n % 2 == 0:
                                introns.append(exons[n] - 1)
                            elif n % 2 != 0:
                                introns.append(exons[n] + 1)
                        else:
                            if exons[n] != la - 1:
                                introns.append(la - 1)
                    for lnc in coords_lnc.keys():
                        coords_in_exon = []
                        coords_in_intron = []
                        for coord in coords_lnc[lnc]:
                            for n in range(len(introns) - 1):
                                if introns[n] <= coord <= introns[n + 1] and n % 2 == 0:
                                    coords_in_intron.append(coord)
                            for n in range(len(exons) - 1):
                                if exons[n] <= coord <= exons[n + 1] and n % 2 == 0:
                                    coords_in_exon.append(coord)
                        out += lnc + '\t'
                        if len(coords_in_exon) == 0:
                            out += 'intronic\n'
                        elif len(coords_in_intron) == 0:
                            out += 'exonic\n'
                        else:
                            full_ret = False
                            for c in coords_in_intron:
                                if c in introns and c + 1 not in exons and c - 1 not in exons:
                                    full_ret = True
                            if full_ret:
                                out += 'intron_retention\n'
                            else:
                                out += 'partial_retention\n'
                    if not isfile(fout):
                        with open(fout, 'w') as f_obj:
                            f_obj.write(out)
                    else:
                        with open(fout, 'a') as f_obj:
                            f_obj.write(out)


def make_lists():
    if not isdir(folder):
        mkdir(folder)

    with open(path_tab, 'r') as f_obj:
        lines = f_obj.readlines()
    num = len(lines)
    cnt = 1
    for line in lines:
        done = int(50 * cnt / num)
        stdout.write("\r[%s%s] %s" % ('=' * done, ' ' * (50 - done),
                     str(cnt) + '/' + str(num) + ' ' + str(2 * done) + '% complete'))
        stdout.flush()
        spl = line.split('\t')
        name = spl[1].split('_')  # TCONS_000088_NC_005087.1
        premrna = name[0] + '_' + name[1]
        out_path = folder + premrna + '.txt'
        if not isfile(out_path):
            with open(out_path, 'w') as f_obj:
                f_obj.write(line)
        else:
            with open(out_path, 'a') as f_obj:
                f_obj.write(line)
        cnt += 1
    print('\n')


def find_tr_tr():
    with open(path_tab, 'r') as f_obj:
        lines = f_obj.readlines()
    out = ''
    names = []
    for line in lines:
        spl = line.split('\t')
        info = spl[1]
        info_spl = info.split(',')  # TCONS_000088_NC_005087.1,0:4157,4811:6827
        if info[:12] == spl[6]:
            out += line
    with open('tr_tr.txt', 'w') as f_obj:
        f_obj.write(out)


def overlap():
    with open(path_tab, 'r') as f_obj:
        lines = f_obj.readlines()
    out = []
    for line in lines:
        spl = line.split('\t')
        info = spl[1]
        pairs = info.split(',')  # TCONS_000088_NC_005087.1,0:4157,4811:6827
        coords = []
        for n in pairs[1:]:
            coord = n.split(':')
            coords.append(int(coord[0]))
            coords.append(int(coord[1]))
        start = int(spl[2])
        end = start + int(spl[3])
        for n in range(len(coords) - 1):
            if coords[n] <= start + D <= coords[n+1]:
                if not (coords[n] <= start + D <= coords[n + 1] and coords[n] <= end - D <= coords[n + 1]):
                    out.append(line)
                    break
        for n in range(1, len(coords)):
            if coords[n-1] <= end - D <= coords[n]:
                if not (coords[n-1] <= start + D <= coords[n] and coords[n-1] <= end - D <= coords[n]):
                    out.append(line)
                    break
    sout = set(out)
    fout = ''
    for n in sout:
        fout += n
    with open('overlap_tab1.txt', 'w') as f_obj:
        f_obj.write(fout)


if __name__ == '__main__':
    find_retention()
    # make_lists()
