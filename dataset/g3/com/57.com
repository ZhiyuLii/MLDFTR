%chk=57.chk
%nproc=16
#p freq pbepbe/6-311++G(3df,3pd) 

G3 HOF calculation for No.57

 0,1
  F                                                 2.353877523400      0.000000000000     -0.985922929800
  C                                                 1.183810388500      0.000000000000     -0.304433211900
  C                                                -1.213791631200      0.000000000000      1.083770560500
  C                                                 0.000000000000      0.000000000000     -1.029069323600
  C                                                 1.213791631200      0.000000000000      1.083770560500
  C                                                 0.000000000000      0.000000000000      1.770604785800
  C                                                -1.183810388500      0.000000000000     -0.304433211900
  H                                                 0.000000000000      0.000000000000     -2.113126116800
  H                                                 2.167338063100      0.000000000000      1.600847540100
  H                                                 0.000000000000      0.000000000000      2.856782817200
  F                                                -2.353877523400      0.000000000000     -0.985922929800
  H                                                -2.167338063100      0.000000000000      1.600847540100

 1 2 1.000
 2 1 1.000 4 1.500 5 1.500
 3 6 1.500 7 1.500 12 1.000
 4 2 1.500 7 1.500 8 1.000
 5 2 1.500 6 1.500 9 1.000
 6 3 1.500 5 1.500 10 1.000
 7 3 1.500 4 1.500 11 1.000
 8 4 1.000
 9 5 1.000
 10 6 1.000
 11 7 1.000
 12 3 1.000
 
