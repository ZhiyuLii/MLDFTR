%chk=58.chk
%nproc=16
#p freq pbepbe/6-311++G(3df,3pd) 

G3 HOF calculation for No.58

 0,1
  C                                                 0.000000000000      0.000000000000      1.367638764500
  C                                                 1.216164323200      0.000000000000      0.696980980300
  C                                                 1.216164323200      0.000000000000     -0.696980980300
  C                                                 0.000000000000      0.000000000000     -1.367638764500
  C                                                -1.216164323200      0.000000000000     -0.696980980300
  C                                                -1.216164323200      0.000000000000      0.696980980300
  F                                                 0.000000000000      0.000000000000      2.723276366000
  F                                                 0.000000000000      0.000000000000     -2.723276366000
  H                                                 2.141995920400      0.000000000000      1.263041639900
  H                                                 2.141995920400      0.000000000000     -1.263041639900
  H                                                -2.141995920400      0.000000000000     -1.263041639900
  H                                                -2.141995920400      0.000000000000      1.263041639900

 1 2 1.500 6 1.500 7 1.000
 2 1 1.500 3 1.500 9 1.000
 3 2 1.500 4 1.500 10 1.000
 4 3 1.500 5 1.500 8 1.000
 5 4 1.500 6 1.500 11 1.000
 6 1 1.500 5 1.500 12 1.000
 7 1 1.000
 8 4 1.000
 9 2 1.000
 10 3 1.000
 11 5 1.000
 12 6 1.000
 
