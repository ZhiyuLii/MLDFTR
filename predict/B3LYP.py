from pyscf import gto
from pyscf import dft
from pyscf import scf
import numpy as np
import pandas as pd
import random


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
mol_num_list = [148, 33, 35, 50, 6 ,88, 75, 40 ,46]

last_name = 'B3LYPTEST/' #the directory to save the data obtained by calculation
first_name = ['g2-AE.txt','g2-anion.txt','g2-aux.txt','g2-cation.txt','g2-extra.txt','g2-small.txt','g3.txt','HTBH.txt','NHTBH.txt',
              'alk19.txt','bde.txt','bde99.txt','bde-extra.txt']
def calculator1(multi, charge,  path, num, name1,name2):
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
            mlpbe.xc = 'b3lyp'
            mlpbe.grids.level = 5
            mlpbe.max_cycle = 50
            A = mlpbe.kernel()
            pred.loc[ii, 'y'] = A * 627.5095
            print("Molecule No.", ii + 1, " converged: ", mlpbe.converged, pred.loc[ii, 'y'])
            ii += 1
        else:
            mlpbe = dft.UKS(mol1)
            mlpbe.xc = 'b3lyp'
            mlpbe.grids.level = 5
            mlpbe.max_cycle = 50
            A = mlpbe.kernel()
            pred.loc[ii, 'y'] = A * 627.5095
            print("Molecule No.", ii + 1, " converged: ", mlpbe.converged, pred.loc[ii, 'y'])
            ii += 1
    pred['y'].to_csv(outputname,sep='\t',header=False,index=False)

def calculator2(multi, charge,  path, spec,name1, name2):
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
            mlpbe.xc = 'b3lyp'
            mlpbe.grids.level = 5
            mlpbe.max_cycle = 50
            A = mlpbe.kernel()
            pred.loc[ii, 'y'] = A * 627.5095
            print("Molecule No.", ii + 1, " converged: ", mlpbe.converged, pred.loc[ii, 'y'])
            ii += 1
        else:
            mlpbe = dft.UKS(mol1)
            mlpbe.xc = 'b3lyp'
            mlpbe.grids.level = 5
            mlpbe.max_cycle = 50
            A = mlpbe.kernel()
            pred.loc[ii, 'y'] = A * 627.5095
            print("Molecule No.", ii + 1, " converged: ", mlpbe.converged, pred.loc[ii, 'y'])
            ii += 1
    pred['y'].to_csv(outputname, sep='\t', header=False, index=False)

for i in range(9):
    calculator1(file_list_multi[i], file_list_charge[i], mol_path_list[i], mol_num_list[i], last_name, first_name[i])
for i in range(4):
    calculator2(file_list_multi[i+9], file_list_charge[i+9], mol_path_list[i+9], file_list_spicies[i], last_name, first_name[i+9])