%chk=41.chk
%nproc=16
#p freq pbepbe/6-311++G(3df,3pd) 

G3 HOF calculation for No.41

 0,1
  H                                                -0.741263521600     -2.074490260400     -0.324013278300
  H                                                -2.139777485700      0.000000000000     -0.111300942200
  C                                                -0.446034430000      1.150136484400      0.181011346200
  C                                                 1.021651840500      0.773897083600     -0.055235568000
  C                                                 1.021651840500     -0.773897083600     -0.055235568000
  C                                                -0.446034430000     -1.150136484400      0.181011346200
  N                                                -1.155644792600      0.000000000000     -0.379469243500
  H                                                -0.630349220400      1.278127175000      1.262512507000
  H                                                -0.741263521600      2.074490260400     -0.324013278300
  H                                                 1.356987992300      1.156504240700     -1.021932890100
  H                                                 1.675565803800      1.195698718600      0.712571815700
  H                                                 1.675565803800     -1.195698718600      0.712571815700
  H                                                 1.356987992300     -1.156504240700     -1.021932890100
  H                                                -0.630349220400     -1.278127175000      1.262512507000

 1 6 1.000
 2 7 1.000
 3 4 1.000 7 1.000 8 1.000 9 1.000
 4 3 1.000 5 1.000 10 1.000 11 1.000
 5 4 1.000 6 1.000 12 1.000 13 1.000
 6 1 1.000 5 1.000 7 1.000 14 1.000
 7 2 1.000 3 1.000 6 1.000
 8 3 1.000
 9 3 1.000
 10 4 1.000
 11 4 1.000
 12 5 1.000
 13 5 1.000
 14 6 1.000
 
