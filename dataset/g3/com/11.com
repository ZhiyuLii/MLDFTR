%chk=11.chk
%nproc=16
#p freq pbepbe/6-311++G(3df,3pd) 

G3 HOF calculation for No.11

 0,1
  C                                                -0.009688000000      0.912760000000      0.000000000000
  C                                                -0.006861000000      0.195983000000      1.201550000000
  C                                                -0.006861000000      0.195983000000     -1.201550000000
  C                                                -0.006861000000     -1.198328000000      1.204953000000
  C                                                -0.006861000000     -1.198328000000     -1.204953000000
  C                                                -0.004765000000     -1.900643000000      0.000000000000
  C                                                 0.029546000000      2.417694000000      0.000000000000
  H                                                 1.060813000000      2.786438000000      0.000000000000
  H                                                -0.467164000000      2.824939000000     -0.884877000000
  H                                                -0.467164000000      2.824939000000      0.884877000000
  H                                                -0.009740000000     -1.736697000000      2.149654000000
  H                                                -0.009740000000     -1.736697000000     -2.149654000000
  H                                                -0.007487000000     -2.987617000000      0.000000000000
  H                                                -0.012712000000      0.736985000000      2.146302000000
  H                                                -0.012712000000      0.736985000000     -2.146302000000

 1 2 1.500 3 1.500 7 1.000
 2 1 1.500 4 1.500 14 1.000
 3 1 1.500 5 1.500 15 1.000
 4 2 1.500 6 1.500 11 1.000
 5 3 1.500 6 1.500 12 1.000
 6 4 1.500 5 1.500 13 1.000
 7 1 1.000 8 1.000 9 1.000 10 1.000
 8 7 1.000
 9 7 1.000
 10 7 1.000
 11 4 1.000
 12 5 1.000
 13 6 1.000
 14 2 1.000
 15 3 1.000
 
