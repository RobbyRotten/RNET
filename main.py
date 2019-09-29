import sys
import math
import pandas as pd
import numpy as np
import os

from parser import Parser
from net_v2 import Nnet


def main():
    save = False
    args = sys.argv
    if '-h' in args or '--help' in args:
        print('\
-Usage for training features extraction, model training and prediction:' +
              '\n\t' + args[0] +
              ' -ex -tr -pr -cd <coding_rna.fasta>\n\
                            -nc <noncoding_rna.fasta> \n\
                            -qr <query_rna.fasta>\n\
                            -sv flag if you want to save feature tables\n\n\
-Usage for model training and prediction:' +
              '\n\t' + args[0] +
              ' -tr -pr -cd <coding_features.csv> <coding_rscu.csv> <coding_hex_freq.csv> \n\
                        -nc <noncoding_features.csv> <coding_rscu.csv> <coding_hex_freq.csv> \n\
                        -qr <query_rna.fasta>\n\
                        -sv flag if you want to save feature tables\n\n\
-Usage for prediction with a trained model:' +
              '\n\t' + args[0] +
              ' -pr -md <model_name>, "model/model" by default\n\
                    -qr <query_rna.fasta>\n\
                    -sv flag if you want to save feature tables')
        exit(0)

    # Training features extraction, training and prediction
    elif '-ex' in args and '-tr' in args and '-pr' in args and \
         '-cd' in args and '-nc' in args and '-qr' in args:
        if '-sv' in args:
            save = True
        cd_file = args[args.index('-cd') + 1]
        nc_file = args[args.index('-nc') + 1]
        qr_file = args[args.index('-qr') + 1]
        if '.fasta' in cd_file and '.fasta' in nc_file and '.fasta' in qr_file:
            # Parsing query file
            print('-Processing ' + qr_file + '...')
            parser_qr = Parser(qr_file)
            parser_qr.parse()
            qr_feat = parser_qr.gen_feat_tab()
            qr_rscu = parser_qr.rscu_tab(save)
            hex_in_qr = parser_qr.count_hex()

            # Parsing ref coding file
            print('-Processing ' + cd_file + '...')
            parser_cd = Parser(cd_file)
            parser_cd.parse()
            cd_feat = parser_cd.gen_feat_tab()
            cd_rscu = parser_cd.rscu_tab(save)
            cd_hex_freq = parser_cd.gen_hex_tab()
            hex_in_cd = parser_cd.count_hex()

            # Parsing ref noncoding file
            print('-Processing ' + nc_file + '...')
            parser_nc = Parser(nc_file)
            parser_nc.parse()
            nc_feat = parser_nc.gen_feat_tab()
            nc_rscu = parser_nc.rscu_tab(save)
            nc_hex_freq = parser_nc.gen_hex_tab()
            hex_in_nc = parser_nc.count_hex()

            h_score_nc = hex_score(cd_hex_freq, nc_hex_freq, hex_in_nc)
            h_score_cd = hex_score(cd_hex_freq, nc_hex_freq, hex_in_cd)
            h_score_qr = hex_score(cd_hex_freq, nc_hex_freq, hex_in_qr)
            h_sc_nc_df = pd.DataFrame([nc_feat['Name'], h_score_nc]).T
            h_sc_cd_df = pd.DataFrame([cd_feat['Name'], h_score_cd]).T
            h_sc_qr_df = pd.DataFrame([qr_feat['Name'], h_score_qr]).T
            del nc_hex_freq, cd_hex_freq, hex_in_nc, hex_in_cd, hex_in_qr
            del h_score_cd, h_score_nc, h_score_qr

            for df in [h_sc_nc_df, h_sc_cd_df, h_sc_qr_df]:
                df.columns = ['Name', 'Hex_score']
            cd_feat = pd.merge(cd_feat, h_sc_cd_df, on='Name')
            nc_feat = pd.merge(nc_feat, h_sc_nc_df, on='Name')
            qr_feat = pd.merge(qr_feat, h_sc_qr_df, on='Name')
            query_names = qr_feat.Name
            del h_sc_cd_df, h_sc_nc_df, h_sc_qr_df
            if save:
                dfs = [cd_feat, nc_feat, qr_feat]
                files = [cd_file, nc_file, qr_file]
                for n in range(3):
                    dfs[n].to_csv('%s_features.csv' % files[n][:-6], sep=";")

            nnet = Nnet(data_cd=cd_feat.drop(['Name'], axis=1).fillna(0),
                        data_nc=nc_feat.drop(['Name'], axis=1).fillna(0),
                        data_qr=qr_feat.drop(['Name'], axis=1).fillna(0),
                        layers=7,
                        epochs=10)
            del cd_feat, nc_feat, qr_feat, qr_rscu, nc_rscu, cd_rscu
            nnet.preprocessing()
            nnet.set_model()
            nnet.train()
            labels_out = nnet.predict()
            nnet.save_model()

            res = ''
            for n in range(len(labels_out)):
                res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
            with open("prediction.txt", 'w') as f_obj:
                f_obj.write(res)
            exit(0)
        else:
            print('Error: coding/noncoding/query fasta file missing or wrong file format')
            exit(1)

    # Training on extracted features and prediction
    elif '-tr' in args and '-pr' in args and '-cd' in args and \
                           '-nc' in args and '-qr' in args:
        if '-sv' in args:
            save = True
        cd_file_f = args[args.index('-cd') + 1]
        cd_file_u = args[args.index('-cd') + 2]
        cd_file_h = args[args.index('-cd') + 3]

        nc_file_f = args[args.index('-nc') + 1]
        nc_file_u = args[args.index('-nc') + 2]
        nc_file_h = args[args.index('-nc') + 3]

        qr_file = args[args.index('-qr') + 1]
        if '.csv' in cd_file_f and '.csv' in nc_file_f and \
           '.csv' in cd_file_u and '.csv' in nc_file_u and '.fasta' in qr_file:

            # Parsing query file
            print('-Processing ' + qr_file + '...')
            parser_qr = Parser(qr_file)
            parser_qr.parse()
            qr_feat = parser_qr.gen_feat_tab()
            qr_rscu = parser_qr.rscu_tab(save)

            cd_feat = pd.read_csv(cd_file_f, sep=';')
            cd_rscu = pd.read_csv(cd_file_u, sep=';')
            cd_hex_freq = pd.read_csv(cd_file_h, sep=';').T
            cd_hex_freq.columns = cd_hex_freq.iloc[0]
            cd_hex_freq = cd_hex_freq.drop(cd_hex_freq.index[0])

            nc_feat = pd.read_csv(nc_file_f, sep=';')
            nc_rscu = pd.read_csv(nc_file_u, sep=';')
            nc_hex_freq = pd.read_csv(nc_file_h, sep=';').T
            nc_hex_freq.columns = nc_hex_freq.iloc[0]
            nc_hex_freq = nc_hex_freq.drop(nc_hex_freq.index[0])

            hex_in_qr = parser_qr.count_hex()
            h_score_qr = hex_score(cd_hex_freq, nc_hex_freq, hex_in_qr)
            h_sc_qr_df = pd.DataFrame([qr_feat['Name'], h_score_qr]).T

            h_sc_qr_df.columns = ['Name', 'Hex_score']
            qr_feat = pd.merge(qr_feat, h_sc_qr_df, on='Name')
            query_names = qr_feat.Name
            del nc_hex_freq, cd_hex_freq, hex_in_qr, h_score_qr, h_sc_qr_df
            if save:
                qr_feat.to_csv('%s_features.csv' % qr_file[:-6], sep=";")

            nnet = Nnet(data_cd=cd_feat.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0),
                        data_nc=nc_feat.drop(['Unnamed: 0', 'Name'], axis=1).fillna(0),
                        data_qr=qr_feat.drop(['Name'], axis=1).fillna(0),
                        layers=7,
                        epochs=10)
            del qr_feat, nc_feat, cd_feat, qr_rscu, nc_rscu, cd_rscu
            nnet.preprocessing()
            nnet.set_model()
            nnet.train()
            labels_out = nnet.predict()
            nnet.save_model()

            res = ''
            for n in range(len(labels_out)):
                res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
            with open("prediction.txt", 'w') as f_obj:
                f_obj.write(res)
            exit(0)
        else:
            print('Error: coding/noncoding features/rscu/hexamer file or query fasta file missing or wrong file format')
            exit(1)

    # Prediction with trained model
    elif '-pr' in args and '-qr' in args and '-md' in args and \
         '-cd' in args and '-nc' in args:
        if '-sv' in args:
            save = True
        qr_file = args[args.index('-qr') + 1]
        model = args[args.index('-qr') + 1]
        cd_file_h = args[args.index('-cd') + 1]
        nc_file_h = args[args.index('-nc') + 1]
        if '.fasta' in qr_file and '.csv' in cd_file_h and '.csv' in nc_file_h:

            # Parsing query file
            print('-Processing ' + qr_file + '...')
            parser_qr = Parser(qr_file)
            parser_qr.parse()
            qr_feat = parser_qr.gen_feat_tab()
            qr_rscu = parser_qr.rscu_tab(save)

            nc_hex_freq = pd.read_csv(nc_file_h, sep=';').T
            nc_hex_freq.columns = nc_hex_freq.iloc[0]
            nc_hex_freq = nc_hex_freq.drop(nc_hex_freq.index[0])

            cd_hex_freq = pd.read_csv(cd_file_h, sep=';').T
            cd_hex_freq.columns = cd_hex_freq.iloc[0]
            cd_hex_freq = cd_hex_freq.drop(cd_hex_freq.index[0])

            hex_in_qr = parser_qr.count_hex()
            h_score_qr = hex_score(cd_hex_freq, nc_hex_freq, hex_in_qr)
            h_sc_qr_df = pd.DataFrame([qr_feat['Name'], h_score_qr]).T

            h_sc_qr_df.columns = ['Name', 'Hex_score']
            qr_feat = pd.merge(qr_feat, h_sc_qr_df, on='Name')
            query_names = qr_feat.Name
            if save:
                qr_feat.to_csv('%s_features.csv' % qr_file[:-6], sep=";")

            nnet = Nnet(data_qr=qr_feat.drop(['Name'], axis=1).fillna(0),
                        layers=7,
                        epochs=10000,
                        model=model)
            nnet.preprocessing()
            nnet.set_model()
            nnet.load_model()
            labels_out = nnet.predict()
            nnet.save_model()

            res = ''
            for n in range(len(labels_out)):
                res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
            with open("prediction.txt", 'w') as f_obj:
                f_obj.write(res)
            nnet.preprocessing()
            nnet.set_model()
            # nnet.train()
            nnet.load_model()
            labels_out = nnet.predict()
            nnet.save_model()

            res = ''
            for n in range(len(labels_out)):
                res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
            with open("prediction.txt", 'w') as f_obj:
                f_obj.write(res)
            exit(0)
        else:
            print('Error: coding/noncoding hexamer file or query fasta file missing or wrong file format')
            exit(1)

    else:
        print('Error: unknown argument')
        exit(1)


def hex_score(ref_cd, ref_nc, query):
    """counts hexamer score for every
       sequence in the query list
       using reference frequencies.
    """
    if str(type(ref_cd)) == "<class 'pandas.core.frame.DataFrame'>" and \
       str(type(ref_nc)) == "<class 'pandas.core.frame.DataFrame'>":
        scores = []
        for seq in query:
            score = 0
            observ_hex = len(query)
            for hexamer in seq:
                if int(ref_cd[hexamer]) == 0:
                    score += -1
                elif int(ref_nc[hexamer]) == 0:
                    score += 1
                else:
                    score += math.log10(int(ref_cd[hexamer]) / int(ref_nc[hexamer]))
            if observ_hex != 0:
                score /= observ_hex
            scores.append(score)
        return pd.Series(scores)
    else:
        scores = []
        for seq in query:
            score = 0
            observ_hex = len(query)
            for hexamer in seq:
                if int(ref_cd[hexamer]) == 0:
                    score += -1
                elif int(ref_nc[hexamer]) == 0:
                    score += 1
                else:
                    score += math.log10(int(ref_cd[hexamer]) / int(ref_nc[hexamer]))
            if observ_hex != 0:
                score /= observ_hex
            scores.append(score)
        return pd.Series(scores)


if __name__ == "__main__":
    main()
