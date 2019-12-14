from net_v3 import Nnet
import numpy as np
n = Nnet()
n.set_model()
inp = np.array([n for n in range(12)])
f_dic = n.forward_pass(inp)
outp = f_dic['layer_6']
target = np.array([0.5])

'''
print(inp.shape, '\n')
for k, v in n.model.items():
    print(k, v.shape)
print('')
for k, v in f_dic.items():
    print(k, v.shape)
'''
b_dic = n.backward_pass(outp, target, f_dic)
for k, v in b_dic.items():
    print(k, v.shape)
