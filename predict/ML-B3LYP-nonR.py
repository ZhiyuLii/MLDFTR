from pyscf import gto
from pyscf import dft
from pyscf import scf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from snn_3h import SNN
import pandas as pd
import random
import torch.nn.functional as F
from sko.SA import SA
import matplotlib.pyplot as plt


input_dim = 3  # rho zeta  r_s
input_dim1 = 4
output_dim = 1
depth = 4
lamda = 1e-5
beta = 1.5
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
hidden1 = [30, 30, 30]
scale_factor = 0.001
iterindex = 0
new_index = 0
scfloss = []
hidden = [30, 30, 30]

s_nn = SNN(input_dim, output_dim, hidden, lamda, beta, use_cuda)
model_path = "nn1.pth"
s_nn.load_state_dict(torch.load(model_path))
s_nn.to(device)



def eval_xc_ml(xc_code, rho, spin, relativity=0, deriv=1, verbose=None):
    if spin == 0:
        rho0, dx, dy, dz = rho[:4]
        gamma1 = gamma2 = gamma12 = (dx ** 2 + dy ** 2 + dz ** 2) * 0.25 + 1e-10
        rho01 = rho02 = rho0 * 0.5
    else:
        rho1 = rho[0]
        rho2 = rho[1]
        rho01, dx1, dy1, dz1 = rho1[:4]
        rho02, dx2, dy2, dz2 = rho2[:4]
        gamma1 = dx1 ** 2 + dy1 ** 2 + dz1 ** 2 + 1e-10
        gamma2 = dx2 ** 2 + dy2 ** 2 + dz2 ** 2 + 1e-10
        gamma12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2 + 1e-10

    rhos = rho01 + rho02  # [grids,][0]=grids
    N = rhos.shape[0]  # list[0]=grids
    ml_in_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)), gamma1.reshape((-1, 1)),
                             gamma2.reshape((-1, 1)), gamma12.reshape((-1, 1))),
                            axis=1)
    ml_in = torch.Tensor(ml_in_)
    ml_in.requires_grad = True
    num_r = ml_in.shape[0]
    dim_in = ml_in.shape[1]
    exc_ml_out = s_nn(ml_in, is_training_data=False)
    ml_exc = exc_ml_out.detach().numpy()
    uni = np.power(rhos, 1 / 3) * 0.75 * np.power(3 / np.pi, 1 / 3)  # (grids,)
    ml_exc = ml_exc * (uni.reshape(N, 1))
    exc_ml = torch.dot(exc_ml_out[:, 0],
                       torch.pow((ml_in[:, 0] + ml_in[:, 1]), 4 / 3) * 0.75 * np.power(3 / np.pi, 1 / 3))
    exc_ml.backward()
    grad = ml_in.grad.data.numpy()  # grid[:,5] for v_r_rho should be held by artificial action as same as 6 and 7

    if spin != 0:
        vrho_ml = np.hstack(((grad[:, 0]).reshape((-1, 1)), (grad[:, 1]).reshape((-1, 1))))
        vgamma_ml = np.hstack((grad[:, 2].reshape((-1, 1)), grad[:, 4].reshape((-1, 1)), grad[:, 3].reshape((-1, 1))))
        vlapl = np.zeros((N, 2))
        vtau = np.zeros((N, 2))
    else:
        vrho_ml = (grad[:, 0] + grad[:, 1]) * 0.5
        vgamma_ml = (grad[:, 2] + grad[:, 3] + grad[:, 4]) * 0.25
        vlapl = np.zeros(N)
        vtau = np.zeros(N)
    b3lyp_xc = dft.libxc.eval_xc('B3LYP', rho, spin, relativity, deriv, verbose)
    b3lyp_exc = np.array(b3lyp_xc[0])
    b3lyp_vrho = np.array(b3lyp_xc[1][0])
    b3lyp_vgamma = np.array(b3lyp_xc[1][1])
    exc = (b3lyp_exc.reshape(-1, 1) + ml_exc).reshape(-1)
    vrho = b3lyp_vrho + vrho_ml
    vgamma = b3lyp_vgamma + vgamma_ml
    vxc = (vrho, vgamma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc



def ml_in(rho,n):  # creator of input_tensor_rhop   将rho转化成input的tensor形式，实际上会组装成一维的tensor，在网络里会继续变形，变成[grids,3]的形状。3代表着 映射的三个参数
    if n==0:
        rho0, dx, dy, dz = rho[:4]
        gamma1 = gamma2 = gamma12 = (dx ** 2 + dy ** 2 + dz ** 2) * 0.25 + 1e-10
        rho01 = rho02 = rho0 * 0.5
    else:
        rho1 = rho[0]
        rho2 = rho[1]
        rho01, dx1, dy1, dz1 = rho1[:4]
        rho02, dx2, dy2, dz2 = rho2[:4]
        gamma1 = dx1 ** 2 + dy1 ** 2 + dz1 ** 2 + 1e-10
        gamma2 = dx2 ** 2 + dy2 ** 2 + dz2 ** 2 + 1e-10
        gamma12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2 + 1e-10
    ml_in_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)), gamma1.reshape((-1, 1)),
                             gamma2.reshape((-1, 1)), gamma12.reshape((-1, 1))), axis=1)
    return ml_in_  # , num_r, dim_in



file_list_multi=['../dataset/g2-AE/multi.txt', '../dataset/g2-anion/multi.txt','../dataset/g2-aux/multi.txt',
                 '../dataset/g2-cation/multi.txt','../dataset/g2-extra/multi.txt','../dataset/g2-small/multi.txt',
                 '../dataset/g3/multi.txt','../dataset/HTBH/multi.txt','../dataset/NHTBH/multi.txt',
                 '../dataset/alk19/multi.txt','../dataset/BDE/multi.txt','../dataset/bde99/multi.txt','../dataset/BDE-extra/multi.txt']  #str
file_list_charge = ['../dataset/g2-AE/charge.txt','../dataset/g2-anion/charge.txt','../dataset/g2-aux/charge.txt',
                    '../dataset/g2-cation/charge.txt','../dataset/g2-extra/charge.txt','../dataset/g2-small/charge.txt',
                    '../dataset/g3/charge.txt','../dataset/HTBH/charge.txt','../dataset/NHTBH/charge.txt',
                    '../dataset/alk19/charge.txt','../dataset/BDE/charge.txt','../dataset/bde99/charge.txt','../dataset/BDE-extra/charge.txt']  # str
file_list_spicies = ['../dataset/alk19/species', '../dataset/BDE/species','../dataset/bde99/species','../dataset/BDE-extra/species.txt'] # str
mol_path_list = ['../dataset/g2-AE/','../dataset/g2-anion/','../dataset/g2-aux/','../dataset/g2-cation/',
                 '../dataset/g2-extra/','../dataset/g2-small/','../dataset/g3/','../dataset/HTBH/','../dataset/NHTBH/',
                 '../dataset/alk19/','../dataset/BDE/','../dataset/bde99/','../dataset/BDE-extra/'] # str
mol_num_list = [148, 33, 35, 50,6,88, 75,40,46]
last_name = 'GGA828-16646/'
first_name = ['g2-AE.txt','g2-anion.txt','g2-aux.txt','g2-cation.txt','g2-extra.txt','g2-small.txt','g3.txt','HTBH.txt','NHTBH.txt',
              'alk19.txt','bde.txt','bde99.txt','bde-extra.txt']


def calculator1(multi, charge,  path, num, name1,name2):
    s_nn.eval()
    pred = pd.DataFrame([], columns=['spin', 'charge', 'y'])
    pred['spin'] = pd.read_table(multi, header=None)
    pred['spin'] = pred['spin'] - 1
    pred['charge'] = pd.read_table(charge, header=None)
    outputname = name1+name2  # str
    mol_path = path
    ii = 0
    print("mol calculation starts")
    for i in range(num):
        mol_file = mol_path+'{}.xyz'.format(1+i)
        mol1         = gto.Mole()
        mol1.verbose = 1
        mol1.atom    = open(mol_file)
        mol1.charge  = int(pred.loc[ii,'charge'])
        mol1.spin    = int(pred.loc[ii,'spin'])
        mol1.basis   = "def2-tzvpd"
        mol1.build()

        if mol1.spin == 0:
            mlpbe = dft.RKS(mol1)
            mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
            mlpbe.grids.level = 5
            mlpbe.max_cycle = 50
            A = mlpbe.kernel()
            print('calculation is down')
            pred.loc[ii, 'y'] = A  * 627.5095
            print("Molecule No.", ii + 1, " converged: ", mlpbe.converged, pred.loc[ii, 'y'])
            ii += 1
        else:
            mlpbe = dft.UKS(mol1)
            mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
            mlpbe.grids.level = 5
            mlpbe.max_cycle = 50
            A = mlpbe.kernel()
            print('calculation is down')
            pred.loc[ii, 'y'] = A * 627.5095
            print("Molecule No.", ii + 1, " converged: ", mlpbe.converged, pred.loc[ii, 'y'])
            ii += 1
    pred['y'].to_csv(outputname,sep='\t',header=False,index=False)

def calculator2(multi, charge,  path, spec,name1, name2):
    s_nn.eval()
    pred = pd.DataFrame([], columns=['spin', 'charge', 'y'])
    pred['spin'] = pd.read_table(multi, header=None)
    pred['spin'] = pred['spin'] - 1
    pred['charge'] = pd.read_table(charge, header=None)
    outputname = name1 + name2  # str
    mol_path = path
    ii = 0

    print("mol calculation starts")
    myfile = open(spec, 'r')
    mycontent = myfile.readlines()
    for i in mycontent:
        mol_file = mol_path + i
        mol_file = mol_file[:len(mol_file) - 1] + '.xyz'
        mol1 = gto.Mole()
        mol1.verbose = 1
        mol1.atom = open(mol_file)
        mol1.charge = int(pred.loc[ii, 'charge'])
        mol1.spin = int(pred.loc[ii, 'spin'])
        mol1.basis = "def2-tzvpd"
        mol1.build()

        if mol1.spin == 0:
            mlpbe = dft.RKS(mol1)
            mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
            mlpbe.grids.level = 5
            mlpbe.max_cycle = 50
            A = mlpbe.kernel()
            print('calculation is down')
            pred.loc[ii, 'y'] = A * 627.5095
            print("Molecule No.", ii + 1, " converged: ", mlpbe.converged, pred.loc[ii, 'y'])
            ii += 1
        else:
            mlpbe = dft.UKS(mol1)
            mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
            mlpbe.grids.level = 5
            mlpbe.max_cycle = 50
            A = mlpbe.kernel()
            print('calculation is down')
            pred.loc[ii, 'y'] = A * 627.5095
            print("Molecule No.", ii + 1, " converged: ", mlpbe.converged, pred.loc[ii, 'y'])
            ii += 1
    pred['y'].to_csv(outputname, sep='\t', header=False, index=False)

for j in range(9):
    calculator1(file_list_multi[j], file_list_charge[j], mol_path_list[j], mol_num_list[j], last_name, first_name[j])
for k in range(4):
    calculator2(file_list_multi[k+9], file_list_charge[k+9], mol_path_list[k+9], file_list_spicies[k], last_name, first_name[k+9])
