import sys
import pandas as pd
import numpy as np
from os.path import isfile, isdir
from os import mkdir
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot

from fasta_parser import Parser
from net_v5 import Nnet


def help(args):
    print('\
-Usage for training features extraction, model training and prediction:' +
          '\n\t' + args[0] +
          ' -ex -tr -pr -cd <coding_rnas.fasta>\n\
                        -nc <noncoding_rnas.fasta> \n\
                        -qr <query_rnas.fasta>\n\
                        -sv flag if you want to save feature tables\n\
                        -sm flag if you want to save a trained model\n\
                        -th number of threads (1 by default)\n\n\
-Usage for model training and prediction:' +
          '\n\t' + args[0] +
          ' -tr -pr -cd <coding_features.csv> <coding_rscu.csv> \n\
                    -nc <noncoding_features.csv> <coding_rscu.csv> \n\
                    -qr <query_rnas.fasta>\n\
                    -sv flag if you want to save feature tables\n\
                    -sm flag if you want to save a trained model\n\
                    -th number of threads (1 by default)\n\n\
-Usage for prediction with a trained model:' +
          '\n\t' + args[0] +
          ' -pr -md <model_name>, "model/model" by default\n\
                -qr <query_rnas.fasta>\n\
                -sv flag if you want to save feature tables\n\
                -th number of threads (1 by default)\n')


def accuracy(args):
    file = args[args.index('-ac') + 1]
    prefix_cd = args[args.index('-cd') + 1]
    prefix_nc = args[args.index('-nc') + 1]

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    with open(file, 'r') as f_obj:
        lines = f_obj.readlines()
    all_lines = len(lines)
    for line in lines:
        line = line.split()
        name = line[0]
        pred = float(line[1])
        if prefix_cd in name and prefix_nc not in name:
            if pred >= 0.5:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if pred < 0.5:
                true_neg += 1
            else:
                false_pos += 1

    true_neg, true_pos, false_neg, false_pos = true_neg / all_lines, true_pos / all_lines, \
                                            false_neg / all_lines, false_pos / all_lines
    print('\
+----------------+------------+\n\
| True positive  |  {:.4}      \n\
+----------------+------------+\n\
| True negative  |  {:.4}      \n\
+----------------+------------+\n\
| False positive |  {:.4}      \n\
-----------------+------------+\n\
| False negative |  {:.4}      \n\
+----------------+------------+\n\
| Total accuracy |  {:.4}      \n\
+----------------+------------+'.format(true_pos, true_neg, false_pos, false_neg, true_pos + true_neg))
    print('positive = coding, negative = noncoding')


def roc(args):
    file = args[args.index('-ro') + 1]
    prefix_cd = args[args.index('-cd') + 1]
    prefix_nc = args[args.index('-nc') + 1]

    probs = []
    testy = []
    with open(file, 'r') as f_obj:
        lines = f_obj.readlines()
    for line in lines:
        line = line.split()
        name = line[0]
        pred = float(line[1])
        if prefix_cd in name and prefix_nc not in name:
            testy.append(1)
            probs.append(pred)
        else:
            testy.append(0)
            probs.append(1-pred)
    auc = roc_auc_score(testy, probs)
    print('AUC: %.3f' % auc)
    fpr, tpr, thresholds = roc_curve(testy, probs)
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlim((-0.05, 1.05))
    pyplot.ylim((-0.05, 1.05))
    pyplot.xlabel('100 - Специфичность')
    pyplot.ylabel('Чувствительность')
    pyplot.grid()
    pyplot.show()


def extract(args, save):
    dfs = []
    files = []
    num = 0
    cd_feat, nc_feat, qr_feat, query_names = None, None, None, None
    if not ('-cd' in args or '-nc' in args or '-qr' in args):
        err(3)

    if '-cd' in args:
        cd_file = args[args.index('-cd') + 1]
        if '.fasta' in cd_file and isfile(cd_file):
            # Parsing ref coding file
            print('-Processing ' + cd_file + '...')
            parser_cd = Parser(cd_file)
            parser_cd.parse()
            cd_feat = parser_cd.gen_feat_tab()
            cd_rscu = parser_cd.rscu_tab(save)
            cd_feat = cd_feat.fillna(0)
            dfs.append(cd_feat)
            files.append(cd_file)
            num += 1
        else:
            err(3, '-cd')

    if '-nc' in args:
        nc_file = args[args.index('-nc') + 1]
        if '.fasta' in nc_file and isfile(nc_file):
            # Parsing ref noncoding file
            print('-Processing ' + nc_file + '...')
            parser_nc = Parser(nc_file)
            parser_nc.parse()
            nc_feat = parser_nc.gen_feat_tab()
            nc_rscu = parser_nc.rscu_tab(save)
            nc_feat = nc_feat.fillna(0)
            dfs.append(nc_feat)
            files.append(nc_file)
            num += 1
        else:
            err(3, '-nc')

    if 'qr' in args:
        qr_file = args[args.index('-qr') + 1]
        if '.fasta' in qr_file and isfile(qr_file):
            # Parsing query file
            print('-Processing ' + qr_file + '...')
            parser_qr = Parser(qr_file)
            parser_qr.parse()
            qr_feat = parser_qr.gen_feat_tab()
            qr_rscu = parser_qr.rscu_tab(save)
            qr_feat = qr_feat.fillna(0)
            query_names = qr_feat.Name
            dfs.append(qr_feat)
            dfs.append(query_names)
            files.append(qr_file)
            num += 1
        else:
            err(3, '-qr')

    if save:
        for n in range(num):
            dfs[n].to_csv('%s_features.csv' % files[n][:-6], sep=";", index=False)

    return {'n': nc_feat, 'c': cd_feat, 'q': qr_feat, 'qn' : query_names}


def read(args, save):
    cd_file_f = args[args.index('-cd') + 1]
    # cd_file_u = args[args.index('-cd') + 2]

    nc_file_f = args[args.index('-nc') + 1]
    # nc_file_u = args[args.index('-nc') + 2]

    qr_file = args[args.index('-qr') + 1]
    if '.csv' in cd_file_f and '.csv' in nc_file_f and '.fasta' in qr_file and \
            isfile(cd_file_f) and isfile(nc_file_f) and isfile(qr_file):

        # Parsing query file
        print('-Processing ' + qr_file + '...')
        parser_qr = Parser(qr_file)
        parser_qr.parse()
        qr_feat = parser_qr.gen_feat_tab()
        # qr_rscu = parser_qr.rscu_tab(save)

        # Reading coding and noncoding files
        cd_feat = pd.read_csv(cd_file_f, sep=';')
        # cd_rscu = pd.read_csv(cd_file_u, sep=';')
        nc_feat = pd.read_csv(nc_file_f, sep=';')
        # nc_rscu = pd.read_csv(nc_file_u, sep=';')

        query_names = qr_feat.Name
        if save:
            qr_feat.to_csv('%s_features.csv' % qr_file[:-6], sep=";", index=False)
        cd_feat = cd_feat.fillna(0)
        nc_feat = nc_feat.fillna(0)
        qr_feat = qr_feat.fillna(0)
        return cd_feat, nc_feat, qr_feat, query_names
    else:
        err(4)


def err(code=10, args=''):
    if code == 1:
        print('Error: coding/noncoding/query fasta file missing')
    elif code == 2:
        print('Error: model object missing')
    elif code == 3:
        print('Error: {}fasta file(s) missing'.format(args))
    elif code == 4:
        print('Error: fasta/csv file(s) missing')
    elif code == 5:
        print('Error: invalid argument "' + args + '"')
    elif code == 6:
        args = args.split()
        print('Error: arguments "' + args[0] + '" and "' + args[1] + ' should not be used together')
    elif code == 7:
        print('Error: argument "' + args + '" argument missing')
    elif code == 8:
        print('Error: argument "' + args + '" is mentioned twice')
    else:
        print('Error')
    exit(1)


def check_args(args, valid_f, valid_a):
    for a in args:
        if a not in valid_f:
            err(5, a)
        if args.count(a) > 1:
            err()
        if a in valid_a and (args[args.index(a) + 1] in valid_a or args[args.index(a) + 1] in valid_f):
            err(7, a)
    if '-tr' in args and '-md' in args:
        err(6, '-tr -md')
    if '-ac' in args:
        ar = ''
        for a in args:
            if a not in ['-ac', '-cd', '-nc']:
                err(6, '-ac {}'.format(ar))


def main():
    sn = str(datetime.now())[5:19].replace('-', '_').replace(' ', '_').replace(':', '_')
    args = sys.argv
    valid_args = ['-md', '-cd', '-nc', '-qr', '-ac']
    valid_flags = ['-h', '--help', '-sv', '-tn', '-ex', '-tr', '-md', '-pr', '-sm', '-cd', '-nc', '-qr', args[0]]
    if '-h' in args or '--help' in args:
        help(args)
        exit(0)
    # check_args(args, valid_args)
    save = False
    threads = 1
    nnet = None
    labels_out = None
    cd_feat, nc_feat, qr_feat, query_names = None, None, None, None

    if '-ro' in args:
        roc(args)
        exit(0)
    if '-ac' in args:
        accuracy(args)
        exit(0)
    if '-sv' in args:
        save = True
    if '-tn' in args:
        threads = args[args.index('-tn') + 1]

    # Training features extraction
    elif '-ex' in args:
        dfs = extract(args, save)
        if '-cd' in args:
            cd_feat = dfs['c']
        elif '-nc' in args:
            nc_feat = dfs['n']
        elif '-qr' in args:
            qr_feat, query_names = dfs['q'], dfs['qn']

    # Training data reading & query features extraction
    elif '-ex' not in args:
        cd_feat, nc_feat, qr_feat, query_names = read(args, save)

    # Model training
    elif '-tr' in args:
        if cd_feat is not None and nc_feat is not None and qr_feat is not None:
            nnet = Nnet(data_cd=cd_feat.drop(['Name'], axis=1).fillna(0),
                        data_nc=nc_feat.drop(['Name'], axis=1).fillna(0),
                        data_qr=qr_feat.drop(['Name'], axis=1).fillna(0),
                        layers_num=5,
                        epochs=10000,
                        threads=threads
                        )
            del cd_feat, nc_feat, qr_feat
            nnet.preprocessing()
            nnet.set_model()
            nnet.train()
        else:
            err(1)

    # Model loading
    elif '-md' in args:
        model = args[args.index('-md') + 1]
        nnet = Nnet(data_cd=cd_feat.drop(['Name'], axis=1).fillna(0),
                    data_nc=nc_feat.drop(['Name'], axis=1).fillna(0),
                    data_qr=qr_feat.drop(['Name'], axis=1).fillna(0),
                    layers_num=5,
                    epochs=10000,
                    threads=threads,
                    model=model
                    )

    # Prediction
    elif '-pr' in args:
        if nnet is not None:
            labels_out = nnet.predict()
            res = ''
            for n in range(len(labels_out)):
                res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
            if not isdir('Predictions'):
                mkdir('Predictions')
            else:
                print('Saving predicton in "Predictions/" ...')
            with open("Predictions/prediction_{}.txt".format(sn), 'w') as f_obj:
                f_obj.write(res)
        else:
            err(2)

    elif '-sm' in args:
        if nnet is not None:
            nnet.save_model()
        else:
            err(2)

    else:
        help(args)

    print('Success')
    exit(0)


if __name__ == "__main__":
    main()
