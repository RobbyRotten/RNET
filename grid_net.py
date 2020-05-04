from net_v5 import Nnet
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
from os import mkdir
from os.path import isfile, isdir

LAYERS = 4


def roc(args, filename):
    file = args[args.index('-ro') + 1]
    prefix_cd = args[args.index('-cd') + 1]
    prefix_nc = args[args.index('-nc') + 1]

    probs = []
    testy = []
    true_pos = 0
    true_neg = 0
    with open(file, 'r') as f_obj:
        lines = f_obj.readlines()
    all_lines = len(lines)
    for line in lines:
        line = line.split()
        name = line[0]
        pred = float(line[1])
        if prefix_cd in name and prefix_nc not in name:
            testy.append(1)
            probs.append(pred)
            if pred >= 0.5:
                true_pos += 1
        else:
            testy.append(0)
            probs.append(1-pred)
            if pred < 0.5:
                true_neg += 1
    accur = (true_pos + true_neg) / all_lines
    auc = roc_auc_score(testy, probs)
    auc = '%.3f' % auc
    accur = '%.3f' % accur
    filename += '_{}auc_{}acc'.format(auc, accur)

    fpr, tpr, thresholds = roc_curve(testy, probs)
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlim((-0.05, 1.05))
    pyplot.ylim((-0.05, 1.05))
    pyplot.xlabel('100 - Специфичность')
    pyplot.ylabel('Чувствительность')
    pyplot.grid()
    pyplot.savefig(filename + '.png')
    pyplot.cla()
    pyplot.clf()


def grid_step(l, n, e, lr, path_cd, path_nc, path_qr, filename):
    coding = pd.read_csv(path_cd, sep=';')
    coding = coding.drop(['Name'], axis=1).fillna(0)

    noncoding = pd.read_csv(path_nc, sep=';')
    noncoding = noncoding.drop(['Name'], axis=1).fillna(0)

    query = pd.read_csv(path_qr, sep=';')
    query_names = query.Name
    query = query.drop(['Name'], axis=1).fillna(0)

    nnet = Nnet(data_cd=coding, data_nc=noncoding, data_qr=query, layers_num=l, epochs=e, hidden=n, lr=lr)
    del query, coding, noncoding

    nnet.preprocessing()
    nnet.set_model()
    nnet.train()
    labels_out = nnet.predict()

    res = ''
    for n in range(len(labels_out)):
        res += query_names[n] + ' ' + str(labels_out[n]) + '\n'
    with open(filename + ".txt", 'w') as f_obj:
        f_obj.write(res)

    args = ['-ro', filename + '.txt', '-cd', 'Pp3c', '-nc', 'lcl|Ppatens']
    roc(args, filename)


def main():
    path_cd = "../equalDS/Ppatens/CSV/test_train/Ppatens_prot_all_features_train.csv"
    path_nc = "../equalDS/Ppatens/CSV/test_train/Ppatens_lncRNAs_qreenc_features_train.csv"
    path_qr = "../equalDS/Ppatens/CSV/test_train/Ppatens_test_nc_cd.csv"

    l = LAYERS

    if not isdir('Grid{}l'.format(l)):
        mkdir('Grid{}l'.format(l))
    neurons = [50, 75, 100, 125, 150, 175, 200]
    epochs = [2, 5000, 10000, 15000, 20000]
    lrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cnt = 1
    total_cnt = len(neurons) * len(epochs) * len(lrs)
    for e in epochs:
        for lr in lrs:
            for n in neurons:
                percentage = '{:.2}'.format(cnt * 100 / total_cnt)
                filename = "Grid{}l/pred_{}l_{}e_{}n_{}lr".format(l, l, e, n, lr)
                progress = '-Processing: {}/{} neurons num, {}/{} epochs num, {}/{} lrs ({}%)'.format(n,
                                                                                                      max(neurons),
                                                                                                      e,
                                                                                                      max(epochs),
                                                                                                      lr,
                                                                                                      max(lrs),
                                                                                                      percentage)
                progr_file = 'Grid{}l/progress.txt'.format(l)
                if not isfile(progr_file):
                    with open(progr_file, 'w') as f_obj:
                        f_obj.write(progress)
                else:
                    with open(progr_file, 'a') as f_obj:
                        f_obj.write('\n' + progress)
                grid_step(l=l, n=n, e=e, lr=lr, path_cd=path_cd, path_nc=path_nc, path_qr=path_qr, filename=filename)
                cnt += 1


if __name__ == "__main__":
    main()
