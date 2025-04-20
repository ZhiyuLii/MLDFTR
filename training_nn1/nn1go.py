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

input_dim = 3
output_dim = 1
depth = 4
lamda = 1e-5
beta = 1.5
use_cuda = False
# device = torch.device("cuda" if use_cuda else "cpu")   when using GPU4pyscf, it works
hidden = [30, 30, 30]
scale_factor = 0.001
iterindex = 0
new_index = 0
scfloss = []



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
    uni = np.power(rhos, 1/3) * 0.75 * np.power(3/np.pi,1/3) #(grids,)
    ml_exc = ml_exc*(uni.reshape(N,1))
    exc_ml = torch.dot(exc_ml_out[:,0], torch.pow((ml_in[:,0]+ml_in[:,1]),4/3)*0.75*np.power(3/np.pi,1/3))
    exc_ml.backward()
    grad = ml_in.grad.data.numpy()  #grid[:,5] for v_r_rho should be held by artificial action as same as 6 and 7

    if spin != 0:
        vrho_ml = np.hstack(((grad[:, 0] ).reshape((-1, 1)), (grad[:, 1]  ).reshape((-1, 1))))
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




def ml_in(rho,spin):
    if spin != 0:  # uks
        rho1 = rho[0]
        rho2 = rho[1]
        rho01, dx1, dy1, dz1 = rho1[:4]
        rho02, dx2, dy2, dz2 = rho2[:4]
        gamma1 = dx1 ** 2 + dy1 ** 2 + dz1 ** 2 + 1e-10
        gamma2 = dx2 ** 2 + dy2 ** 2 + dz2 ** 2 + 1e-10
        gamma12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2 + 1e-10
    else:  # rks
        rho0, dx, dy, dz = rho[:4]
        gamma1 = gamma2 = gamma12 = (dx ** 2 + dy ** 2 + dz ** 2) * 0.25 + 1e-10
        rho01 = rho02 = rho0 * 0.5
    rhos = rho01 + rho02  # [grids,][0]=grids
    ml_in_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)), gamma1.reshape((-1, 1)),
                             gamma2.reshape((-1, 1)), gamma12.reshape((-1, 1))),
                            axis=1)
    return ml_in_  # , num_r, dim_in


class CustomMSELoss(nn.Module):  # loss生成
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, mollist, output, target_data, myexc, g_weights, rho, mydm, myk, e1, ecoul, enuc):
        myloss = [0] * len(mollist)

        def loss(output, target_data, myexc, g_weights, rho, mydm, myk, e1, ecoul, enuc):
            if mydm.shape[0] != 2:  # RKS
                output += torch.tensor(myexc).unsqueeze(1)  # shape
                myExc = ((output * torch.tensor(g_weights).unsqueeze(1) *
                          torch.tensor(rho[0]).unsqueeze(1)).real).sum()  # change to the func of output
                myEv = np.einsum('ij,ji->', myk, mydm).real * .25
                myExc -= myEv
                mytot = myExc + torch.tensor([ecoul + e1 + enuc])
                mytot *= 627.5095
                theloss = torch.abs(mytot - target_data)  # delta
            else:  # UKS
                output += torch.Tensor(myexc).unsqueeze(1)
                myExc = (output * torch.tensor(g_weights).unsqueeze(1) * torch.tensor(rho[0][0]).unsqueeze(1) +
                         output * torch.tensor(g_weights).unsqueeze(1) * torch.Tensor(rho[1][0]).unsqueeze(1)).sum()
                myEv = (np.einsum('ij,ji->', myk[0], mydm[0]).real +
                        np.einsum('ij,ji->', myk[1], mydm[1]).real) * .5
                myExc -= myEv
                mytot = myExc + torch.tensor([ecoul + e1 + enuc])
                mytot *= 627.5095
                theloss = torch.abs(mytot - target_data)
            return theloss

        tloss = torch.tensor([1e-10])
        for i in range(len(mollist)):
            myloss[i] = loss(output[i], target_data[i], myexc[i], g_weights[i], rho[i], mydm[i], myk[i],
                             e1[i],ecoul[i], enuc[i])  # 在trainingmodel里面也要把output输出成一个tensorlist
            tloss += myloss[i]
        tloss /= len(mollist)
        print('here', tloss)
        return tloss  # return a list in


def renew_rho(mollist, n=-1):  # 0 for best_para to BP, 1~num_particles is for each particle
    global target_data
    global scfloss
    mydm = [0] * len(mollist)
    rho = [0] * len(mollist)
    myk = [0] * len(mollist)
    e1 = [0] * len(mollist)
    ecoul = [0] * len(mollist)
    enuc = [0] * len(mollist)
    myweight = [0] * len(mollist)
    mycoords = [0] * len(mollist)
    myexc_0 = [0] * len(mollist)
    scf_loss = 0
    for i in range(len(mollist)):
        if mollist[i].spin == 0:
            mydft = dft.rks.RKS(mollist[i])
            mydft.grids.level = 5
            mydft = mydft.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
            A = mydft.kernel()
            scf_loss += abs(A * 627.5095 - target_data[i])
            mycoords_ = mydft.grids.coords
            myweight_ = mydft.grids.weights
            myao = dft.numint.eval_ao(mollist[i], mydft.grids.coords, deriv=1)
            mydm_ = mydft.make_rdm1()
            rho_ = dft.numint.eval_rho(mollist[i], myao, mydm_, xctype='GGA')   #???这里应该有问题。和numint中的make_rho是否等价？
            myexc_0_ = dft.libxc.eval_xc('B3LYP', rho_, spin=mollist[i].spin, relativity=0, deriv=1, verbose=None)[0]
            myj, myk_ = mydft.get_jk(mollist[i], mydm_, 1)
            myk_ *= 0.2  # hyb HF of b3lyp, should be delete from xc energy
            e1_ = mydft.scf_summary['e1']  # numpy.float64 can do add-operation with tensor
            ecoul_ = mydft.scf_summary['coul']  # numpy.float64
            enuc_ = mydft.energy_nuc()

        else:
            mydft = dft.uks.UKS(mollist[i])
            mydft.grids.level = 5
            mydft = mydft.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
            A = mydft.kernel()
            scf_loss += abs(A * 627.5095 - target_data[i])
            mycoords_ = mydft.grids.coords
            myweight_ = mydft.grids.weights
            myao = dft.numint.eval_ao(mollist[i], mydft.grids.coords, deriv=1)
            mydm_ = mydft.make_rdm1()
            rhoa = dft.numint.eval_rho(mollist[i], myao, mydm_[0], xctype='GGA')  #???这里应该有问题。
            rhob = dft.numint.eval_rho(mollist[i], myao, mydm_[1], xctype='GGA')
            rho_ = (rhoa, rhob)
            myj, myk_ = mydft.get_jk(mollist[i], mydm_, 1)
            myk_ *= 0.2  # hyb HF of b3lyp, should be delete from xc energy
            e1_ = mydft.scf_summary['e1']  # numpy.float64 can do add-operation with tensor
            ecoul_ = mydft.scf_summary['coul']  # numpy.float64
            enuc_ = mydft.energy_nuc()
            myexc_0_ = dft.libxc.eval_xc('B3LYP', rho_, spin=mollist[i].spin, relativity=0, deriv=1, verbose=None)[0]
        mydm[i] = mydm_
        rho[i] = rho_
        myk[i] = myk_
        e1[i] = e1_
        ecoul[i] = ecoul_
        enuc[i] = enuc_
        myweight[i] = myweight_
        myexc_0[i] = myexc_0_
        mycoords[i] = mycoords_
    scfloss.append(scf_loss / len(mollist))  # 更新的同时，增加下一刻的scf
    print('here renew rho and scfloss is ', scfloss)
    my_dict = {}
    for i in range(len(mollist)):
        my_dict['dm{}'.format(i)] = mydm[i]
        my_dict['rho{}'.format(i)] = rho[i]
        my_dict['myk{}'.format(i)] = myk[i]
        my_dict['e1{}'.format(i)] = e1[i]
        my_dict['ecoul{}'.format(i)] = ecoul[i]
        my_dict['myweight{}'.format(i)] = myweight[i]
        my_dict['myexc_0{}'.format(i)] = myexc_0[i]
        my_dict['enuc{}'.format(i)] = enuc[i]
        my_dict['mycoords{}'.format(i)] = mycoords[i]
    np.savez('npz/mylist{}'.format(n + 1), **my_dict)  # 不是，这个是粒子的序数


########################################################################################
'''atom and mol'''

verbosity = 1
basis = 'def2tzvpd'
df1 = pd.read_csv('g2ref.txt', header=None, names=['REF'])
df2 = pd.read_csv('../dataset/g2-AE/multi.txt', header=None, names=['spin'])
merged_df = pd.concat([df1, df2], axis=1)  #

file_info_list = []
for index, row in merged_df.iterrows():
    file_info = {
        'File': f'{index + 1}.xyz',
        'REF': row['REF'],
        'spin': row['spin']
    }
    file_info_list.append(file_info)


def read_float_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [float(line.strip()) for line in lines]
    return data


random_indices = [25, 86, 65, 134, 14,
                  131, 62, 64, 33, 75,
                  28, 18, 26, 55, 27,
                  85, 81, 94, 15, 32,
                  18, 38, 54, 61, 59,
                  60, 79, 98, 101, 106,
                  136, 143, 39, 40, 41]  # 35 moleculars from g2 set
myfile = open('iterations.txt', 'a+')
myfile.write('start from here and random set is ' + ','.join(map(str, random_indices)) + '\n')
myfile.close()

mollist = []
target_data = []

atompath = '../dataset/atoms/'
atomsl = ['{}.xyz'.format(i + 149) for i in range(15)]
atomsref = [-313.8, -4692.2, -9203.3, -15468.6, -23745.8, -34252.7, -47102.5, -62580.8, -101789.3, -125516.3, -152035.8,
            -181537.2, -214104.5, -249778.1, -288707.5]
mult = [2, 2, 1, 2, 3, 4, 3, 2, 2, 1, 2, 3, 4, 3, 2]
for i in range(15):  # atoms' Mole instance
    mol = gto.Mole()
    mol.verbose = verbosity
    path = atompath + atomsl[i]
    mol.atom = open(path)
    mol.charge = 0
    mol.spin = int(mult[i] - 1)
    mol.basis = basis
    mol.build()
    mollist.append(mol)
    ref = atomsref[i]
    target_data.append(ref)
for _ in random_indices: # moleculars' Mole instance
    mol = gto.Mole()
    mol.verbose = verbosity
    X = file_info_list[_]
    path = '../dataset/g2-AE/' + X['File']
    mol.atom = open(path)
    mol.charge = 0
    mol.spin = int(X['spin'] - 1)
    mol.basis = basis
    mol.build()
    mollist.append(mol)
    ref = X['REF']
    target_data.append(ref)

x = 25
valid_mol = mollist[x:]
valid_ref = target_data[x:]
mollist = mollist[:x]
target_data = target_data[:x]
###################################################################################

s_nn = SNN(input_dim, output_dim, hidden, lamda, beta, use_cuda)
model_path = "gpara16646.pth"   #the best points getting from SO
s_nn.load_state_dict(torch.load(model_path))
# s_nn.to(device)

criterion = CustomMSELoss()


def training_model(model, mollist, target_data,valid_mol, valid_ref, criterion, optimizer, epochs_num=100):
    s_nn.eval()
    renew_rho(mollist)
    for epoch in range(epochs_num):
        optimizer.zero_grad()
        if ((epoch + 1) % 20) == 0:
            s_nn.eval()
            renew_rho(mollist)
            valoss = 0
            for i in range(len(valid_mol)):
                s_nn.eval()
                if valid_mol[i].spin == 0:
                    vadft = dft.rks.RKS(valid_mol[i])
                    vadft.grids.level = 5
                    vadft = vadft.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
                    B = vadft.kernel()
                    valoss += abs(B * 627.5095 - valid_ref[i])
                    print(B, vadft.converged)
                else:
                    vadft = dft.uks.UKS(valid_mol[i])
                    vadft.grids.level = 5
                    vadft = vadft.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
                    B = vadft.kernel()
                    valoss += abs(B * 627.5095 - valid_ref[i])
                    print(B, vadft.converged)
            print('iterations [{}], valoss: {:.4f}'.format(epoch + 1, valoss/len(mollist)))
            thefile = open('nn1go.txt', 'a+')
            thefile.write('iterations [{}], valoss: {:.4f}\n'.format(epoch + 1, float(valoss/len(mollist))))
            thefile.close()
        s_nn.train()
        data = np.load('npz/mylist0.npz')
        dm = [0] * len(mollist)
        rho = [0] * len(mollist)
        e1 = [0] * len(mollist)
        ecoul = [0] * len(mollist)
        enuc = [0] * len(mollist)
        myweight = [0] * len(mollist)
        myexc_0 = [0] * len(mollist)
        mycoords = [0] * len(mollist)
        myk = [0]*len(mollist)
        for i in range(len(mollist)):
            dm[i] = data['dm{}'.format(i)]
            rho[i] = data['rho{}'.format(i)]
            e1[i] = data['e1{}'.format(i)]
            ecoul[i] = data['ecoul{}'.format(i)]
            myk[i] = data['myk{}'.format(i)]
            myweight[i] = data['myweight{}'.format(i)]
            myexc_0[i] = data['myexc_0{}'.format(i)]
            enuc[i] = data['enuc{}'.format(i)]
            mycoords[i] = data['mycoords{}'.format(i)]
        output = [0] * len(mollist)
        rhoml = [0] * len(mollist)
        myinput = [0] * len(mollist)
        for i in range(len(mollist)):
            rhoml[i] = ml_in(rho[i], mollist[i].spin)
            myinput[i] = torch.tensor(rhoml[i], requires_grad=True)
            output[i] = s_nn(myinput[i], is_training_data=False)
            uni = torch.pow(myinput[i][:, 0] + myinput[i][:, 1], 1 / 3) * 0.75 * np.power(3 / np.pi,
                                                                                          1 / 3)  # (grids,)
            output[i] = output[i] * uni.unsqueeze(1)
        Loss = criterion(mollist, output, target_data, myexc_0, myweight, rho, dm, myk,e1, ecoul,
                         enuc)
        print('iterations [{}], valoss: {:.4f}'.format(epoch + 1, Loss.item()))
        thefile = open('nn1go.txt', 'a+')
        thefile.write('iterations [{}], Loss: {:.4f}\n'.format(epoch + 1, Loss.item()))
        thefile.close()
        Loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs_num, Loss.item()))
        torch.save(model.state_dict(), 'nn1go/nn1gopara{}.pth'.format(epoch + 1))
        '''saving the parameters, make sure you have a 'nn1go' directory '''


optimizer = optim.Rprop([
    {'params': s_nn.model[0].parameters(), 'lr': 1e-5},
    {'params': s_nn.model[2].parameters(), 'lr': 1e-5},
    {'params': s_nn.model[4].parameters(), 'lr': 1e-5},
    {'params': s_nn.model[6].parameters(), 'lr': 5e-6}
])  # yield a class for optim, set learning rate

num_epochs = 400   #
training_model(s_nn, mollist, target_data,valid_mol, valid_ref, criterion,optimizer,num_epochs)
