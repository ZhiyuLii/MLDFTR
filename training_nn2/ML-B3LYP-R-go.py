from pyscf import gto
from pyscf import dft
from pyscf import scf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from snn_3h import SNN
from snn_3hR import SNN2
import pandas as pd
import random
import torch.nn.functional as F
from sko.SA import SA
import matplotlib.pyplot as plt
from frequencyselected import k_set_generator
from normal_coef import get_normal_coefficient


input_dim_nn1 = 3  # rho zeta  r_s
input_dim_nn2 = 4  # rho zeta r_s R

output_dim = 1  #same dimension of output
hidden = [30, 30, 30]  #same hidden
depth = 4
lamda = 1e-5
beta = 1.5
use_cuda = False
# device = torch.device("cuda" if use_cuda else "cpu")
scale_factor = 0.001
iterindex = 0
new_index = 0
scfloss = []


'''The weight D of training set, D0 for validation set '''
D = [1,1,10,2,2,2,10,10,10,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
D0 = [1 for i in range(25)]

p = 40
f = 27000
fppath = 'S'+ str(p)+'-'+str(f)+'.npz'
pfname = str(p)+'-'+str(f)
k_set_generator(p, f)
SNc = get_normal_coefficient(p,f )
##########################################################################
'''R has already been calculated during the SO process and 

    does not need to be recalculated, as this is a fully POST-SCF process.'''




#############################################################################################################

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


def ml_in(rho):  #in put of nn1
    if rho.shape[0] != 2:
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


#############################################################################################################
def ml_inR(rho, weight, coords, i, m):   # input of nn2
    if rho.shape[0] == 2:  # uks
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
    if m==0:
        RS = np.load('Rtraining/R{}.npz'.format(i))
    else:
        RS = np.load('Rvalidation/R{}.npz'.format(i))
    R = RS['arr_0']
    ml_in_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)), gamma1.reshape((-1, 1)),
                             gamma2.reshape((-1, 1)), gamma12.reshape((-1, 1)),
                             R.reshape((-1, 1))),
                            axis=1)  # 这里输入了十个R，该张量在用作测试时，仅仅使用了[5:]之后的张量
    return ml_in_  # , num_r, dim_in








#############################################################################################################


class CustomMSELoss(nn.Module):  # loss生成
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, mollist, output, target_data, myexc, g_weights, rho, mydm, myk, e1, ecoul, enuc,DD):
        myloss = [0] * len(mollist)

        def loss(output, target_data, myexc, g_weights, rho, mydm, myk, e1, ecoul, enuc,d):
            if mydm.shape[0] != 2:  # RKS
                output += torch.tensor(myexc).unsqueeze(1)  # shape
                myExc = ((output * torch.tensor(g_weights).unsqueeze(1) *
                          torch.tensor(rho[0]).unsqueeze(1)).real).sum()  # change to the func of output
                myEv = np.einsum('ij,ji->', myk, mydm).real * .25
                myExc -= myEv
                mytot = myExc + torch.tensor([ecoul + e1 + enuc])
                mytot *= 627.5095
                theloss = torch.abs(mytot - target_data)  # delta
                theloss *=d
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
                theloss *=d
            return theloss

        tloss = torch.tensor([1e-10])
        for i in range(len(mollist)):
            myloss[i] = loss(output[i], target_data[i], myexc[i], g_weights[i], rho[i], mydm[i], myk[i],
                             e1[i],ecoul[i], enuc[i],DD[i])  # 在trainingmodel里面也要把output输出成一个tensorlist
            #  print(myloss[i])
            tloss += myloss[i]
        tloss /= len(mollist)
        return tloss  # return a list in

#############################################################################################################

def renew_rho(mollist, n=-1):   # default training.    n = 0 for validation density data
    global target_data
    global scfloss
    global valid_ref
    mydata = 0
    if n==0:
        mydata = valid_ref
    else:
        mydata = target_data
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
            scf_loss += abs(A * 627.5095 - mydata[i])
            print(scf_loss)
            mycoords_ = mydft.grids.coords
            myweight_ = mydft.grids.weights
            myao = dft.numint.eval_ao(mollist[i], mydft.grids.coords, deriv=1)
            mydm_ = mydft.make_rdm1()
            rho_ = dft.numint.eval_rho(mollist[i], myao, mydm_, xctype='GGA')
            myexc_0_ = dft.libxc.eval_xc('B3LYP', rho_, spin=mollist[i].spin, relativity=0, deriv=1, verbose=None)[0]
            myj, myk_ = mydft.get_jk(mollist[i], mydm_, 1)
            myk_ *= 0.2  # hyb HF of b3lyp, should be delete from xc energy
            e1_ = mydft.scf_summary['e1']
            ecoul_ = mydft.scf_summary['coul']
            enuc_ = mydft.energy_nuc()

        else:
            mydft = dft.uks.UKS(mollist[i])
            mydft.grids.level = 5
            mydft = mydft.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])
            A = mydft.kernel()
            scf_loss += abs(A * 627.5095 - mydata[i])
            print(scf_loss)
            mycoords_ = mydft.grids.coords
            myweight_ = mydft.grids.weights
            myao = dft.numint.eval_ao(mollist[i], mydft.grids.coords, deriv=1)
            mydm_ = mydft.make_rdm1()
            rhoa = dft.numint.eval_rho(mollist[i], myao, mydm_[0], xctype='GGA')  #???这里应该有问题。
            rhob = dft.numint.eval_rho(mollist[i], myao, mydm_[1], xctype='GGA')
            rho_ = (rhoa, rhob)
            myj, myk_ = mydft.get_jk(mollist[i], mydm_, 1)
            myk_ *= 0.2  # hyb HF of b3lyp, should be delete from xc energy
            e1_ = mydft.scf_summary['e1']  #
            ecoul_ = mydft.scf_summary['coul']  
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
    np.savez('nn1density/mylist{}'.format(n + 1), **my_dict)  # 不是，这个是粒子的序数


########################################################################################
'''atom and mol'''
verbosity = 1
basis = 'def2tzvpd'
df1 = pd.read_csv('g2ref.txt', header=None, names=['REF'])
df2 = pd.read_csv('../dataset/g2-AE/multi.txt', header=None, names=['spin'])
merged_df = pd.concat([df1, df2], axis=1)  # spin和ref对应

# 打包文件名、大小和属性信息
file_info_list = []
for index, row in merged_df.iterrows():
    file_info = {
        'File': f'{index + 1}.xyz',
        'REF': row['REF'],
        'spin': row['spin']
    }
    file_info_list.append(file_info)

# 上述坐标文件名、spin和ref values通通打包好了。
def read_float_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 每行数据以换行符结尾，可以使用strip()方法去除换行符
    data = [float(line.strip()) for line in lines]
    return data

random_indices = [25,86,65,134,14,131,64,33,28,18,26,55,27,85,81,94,15,32,18,38,54,61,59,60,79,98,101,106,136,143, 39, 41]

mollist = []
target_data = []
########################################
molealk1= '../dataset/alk19/29_hexamethylethane_Alk44.xyz'
molealk2='../dataset/alk19/25_neohexane_Alk44.xyz'
molealk3 = '../dataset/alk19/23_isooctane_Alk44.xyz' 
smol1 = gto.Mole()
smol1.verbose = verbosity
smol1.atom = open(molealk1)
smol1.charge = 0
smol1.spin = 0
smol1.basis = basis
smol1.build()

smol2 = gto.Mole()
smol2.verbose = verbosity
smol2.atom = open(molealk2)
smol2.charge = 0
smol2.spin = 0
smol2.basis = basis
smol2.build()

smol3 = gto.Mole()
smol3.atom = open(molealk3)
smol3.verbose = verbosity
smol3.charge = 0
smol3.spin = 0
smol3.basis = basis
smol3.build()

mollist.append(smol1)
mollist.append(smol2)
target_data.append(-198098.8)
target_data.append(-148763.2)


atompath = '../dataset/atoms/'
atomsl = ['{}.xyz'.format(i+149) for i in range(15)]
atomsref=[-313.8, -4692.2, -9203.3, -15468.6, -23745.8, -34252.7, -47102.5, -62580.8, -101789.3, -125516.3, -152035.8, -181537.2, -214104.5, -249778.1, -288707.5]
mult=[2,2,1,2,3,4,3,2,2,1,2,3,4,3,2]
for i in range(15):
    mol = gto.Mole()
    mol.verbose = verbosity
    path = atompath + atomsl[i]
    mol.atom = open(path)
    mol.charge = 0
    mol.spin = int(mult[i]-1)
    mol.basis = basis
    mol.build()
    mollist.append(mol)
    ref = atomsref[i]
    target_data.append(ref)
for _ in random_indices:
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

mollist.append(smol3)
target_data.append(-198099.1)
x = 25
valid_mol = mollist[x:]
valid_ref = target_data[x:]
mollist = mollist[:x]
target_data = target_data[:x]


#############################################################################################################
s_nn = SNN(input_dim_nn1, output_dim, hidden, lamda, beta, use_cuda)
model_path_nn1 = "nn1.pth"
s_nn.load_state_dict(torch.load(model_path_nn1))
# s_nn.to(device)
criterion = CustomMSELoss()
nn2 = SNN2(input_dim_nn2, output_dim, hidden, lamda, beta, use_cuda)
# nn2.to(device)

model_path_nn2_so = 'nn2so'+pfname+'/para{}.pth'  # the nn2 parameters obtained by SO
nn2.load_state_dict(torch.load(model_path_nn2_so))
# nn2.to(device)

#############################################################################################################

def training_model(model, mollist, target_data, valia_mol, valid_ref, criterion, optimizer, epochs_num=1000):
    for epoch in range(epochs_num):
        optimizer.zero_grad()
        data1 = np.load('nn1density/mylist1.npz')
        dm = [0] * len(valid_mol)
        rho = [0] * len(valid_mol)
        myk = [0] * len(valid_mol)
        e1 = [0] * len(valid_mol)
        ecoul = [0] * len(valid_mol)
        enuc = [0] * len(valid_mol)
        myweight = [0] * len(valid_mol)
        myexc_0 = [0] * len(valid_mol)
        mycoords = [0] * len(valid_mol)
        R = [0] * len(valid_mol)
        for i in range(len(valid_mol)):
            dm[i] = data1['dm{}'.format(i)]
            rho[i] = data1['rho{}'.format(i)]
            myk[i] = data1['myk{}'.format(i)]
            e1[i] = data1['e1{}'.format(i)]
            ecoul[i] = data1['ecoul{}'.format(i)]
            myweight[i] = data1['myweight{}'.format(i)]
            myexc_0[i] = data1['myexc_0{}'.format(i)]
            enuc[i] = data1['enuc{}'.format(i)]
            mycoords[i] = data1['mycoords{}'.format(i)]
        output = [0] * len(mollist)
        rhoml = [0] * len(mollist)
        myinput = [0] * len(mollist)
        output1 = [0] * len(mollist)
        rhoml1 = [0] * len(mollist)
        myinput1 = [0] * len(mollist)
        output2 = [0] * len(mollist)
        for i in range(len(mollist)):
            rhoml[i] = ml_inR(rho[i], myweight[i], mycoords[i], i, 1)
            myinput[i] = torch.tensor(rhoml[i], requires_grad=True)
            output[i] = nn2(myinput[i], is_training_data=False)  # 这部分与参数相关，因此放到不同粒子的循环内部
            rhoml1[i] = ml_in(rho[i])
            myinput1[i] = torch.tensor(rhoml1[i], requires_grad=True)
            output1[i] = s_nn(myinput1[i], is_training_data=False)
            uni = torch.pow(myinput1[i][:, 0] + myinput1[i][:, 1], 1 / 3) * 0.75 * np.power(3 / np.pi,
                                                                                            1 / 3)  # (grids,)
            output2[i] = (output[i] + output1[i]) * uni.unsqueeze(1)
            rhoml[i] = 0.00001
            myinput[i] = 0.00001
            output[i] = 0.00001
            rhoml1[i] = 0.00001
            myinput1[i] = 0.00001
            output1[i] = 0.00001

        VALoss = criterion(valid_mol, output2, valid_ref, myexc_0, myweight, rho, dm, myk, e1, ecoul,
                           enuc,D0)
        print('iterations [{}], valoss: {:.4f}'.format(epoch + 1, VALoss.item()))
        thefile = open('nn2go'+pfname+'.txt', 'a+')
        thefile.write('iterations [{}], valoss: {:.4f}\n'.format(epoch + 1, VALoss.item()))
        thefile.close()
        optimizer.zero_grad()
        nn2.eval()
        s_nn.eval()
        thefile.close()
        nn2.train()
        data = np.load('nn1density/mylist0.npz')
        dm = [0] * len(mollist)
        rho = [0] * len(mollist)
        myk = [0] * len(mollist)
        e1 = [0] * len(mollist)
        ecoul = [0] * len(mollist)
        enuc = [0] * len(mollist)
        myweight = [0] * len(mollist)
        myexc_0 = [0] * len(mollist)
        mycoords = [0] * len(mollist)
        for i in range(len(mollist)):
            dm[i] = data['dm{}'.format(i)]
            rho[i] = data['rho{}'.format(i)]
            myk[i] = data['myk{}'.format(i)]
            e1[i] = data['e1{}'.format(i)]
            ecoul[i] = data['ecoul{}'.format(i)]
            myweight[i] = data['myweight{}'.format(i)]
            myexc_0[i] = data['myexc_0{}'.format(i)]
            enuc[i] = data['enuc{}'.format(i)]
            mycoords[i] = data['mycoords{}'.format(i)]
        output = [0] * len(mollist)
        rhoml = [0] * len(mollist)
        myinput = [0] * len(mollist)
        output1 = [0] * len(mollist)
        rhoml1 = [0] * len(mollist)
        myinput1 = [0] * len(mollist)
        output2 = [0] * len(mollist)
        for i in range(len(mollist)):
            rhoml[i] = ml_inR(rho[i], myweight[i], mycoords[i], i, 0)
            myinput[i] = torch.tensor(rhoml[i], requires_grad=True)
            output[i] = nn2(myinput[i], is_training_data=False)  # 这部分与参数相关，因此放到不同粒子的循环内部
            rhoml1[i] = ml_in(rho[i])
            myinput1[i] = torch.tensor(rhoml1[i], requires_grad=True)
            output1[i] = s_nn(myinput1[i], is_training_data=False)
            uni = torch.pow(myinput1[i][:, 0] + myinput1[i][:, 1], 1 / 3) * 0.75 * np.power(3 / np.pi,
                                                                                            1 / 3)  # (grids,)
            output2[i] = (output[i] + output1[i]) * uni.unsqueeze(1)
            rhoml[i] = 0.00001
            myinput[i] = 0.00001
            output[i] = 0.00001
            rhoml1[i] = 0.00001
            myinput1[i] = 0.00001
            output1[i] = 0.00001

        Loss = criterion(mollist, output2, target_data, myexc_0, myweight, rho, dm, myk, e1, ecoul,
                         enuc,D)
        Loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs_num, Loss.item()))
        thefile = open('nn2so'+pfname+'.txt', 'a+')
        thefile.write('iterations [{}], Loss: {:.4f}\n'.format(epoch + 1, Loss.item()))
        thefile.close()
        torch.save(model.state_dict(), 'nn2go'+pfname+'/model_params{}.pth'.format(epoch + 1))

#############################################################################################################
optimizer = optim.Rprop([
    {'params': nn2.model[0].parameters(), 'lr': 1e-5},
    {'params': nn2.model[2].parameters(), 'lr': 1e-5},
    {'params': nn2.model[4].parameters(), 'lr': 1e-5},
    {'params': nn2.model[6].parameters(), 'lr': 5e-6}
])  # yield a class for optim
num_epochs = 500
training_model(nn2, mollist, target_data, valid_mol, valid_ref, criterion, optimizer, num_epochs)


