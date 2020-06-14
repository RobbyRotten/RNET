import matplotlib.pyplot as plt
from matplotlib import cm, rc
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
import numpy as np

path = '../Grid6l'
files = listdir(path)
pngs = []
for f in files:
    if '.png' in f:
        pngs.append(f)
x, y, z, z1 = [], [], [], []
for png in pngs:
    if 'pred' in png:
        name = png.split('_')
        neurons = name[3][:-1]
        lr = name[4][:-2]
        auc = name[5][:-3]
        acc = name[6][:-7]
        if float(auc) != 1.0:
            x.append(neurons)
            y.append(lr)
            z.append(auc)
            z1.append(acc)

X = np.array(x).astype(float)
Y = np.array(y).astype(float)
Z = np.array(z).astype(float)
Z1 = np.array(z1).astype(float)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Ч-ло нейронов в первом скрытом слое', fontsize=7)
ax.set_ylabel('Коэфф. скор. обучения', fontsize=7)
ax.set_zlabel('AUC', fontsize=7)
plt.title('max(AUC) = ' + str(max(Z)))
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.get_cmap('coolwarm'))
fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.savefig(path + 'plot_auc.png')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Ч-ло нейронов в первом скрытом слое', fontsize=7)
ax.set_ylabel('Коэфф. скор. обучения', fontsize=7)
ax.set_zlabel('Acc', fontsize=7)
plt.title('max(Acc) = ' + str(max(Z1)))
surf = ax.plot_trisurf(X, Y, Z1, cmap=cm.get_cmap('coolwarm'))
fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.savefig(path + 'plot_acc.png')
