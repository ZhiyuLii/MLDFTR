%chk=21.chk
%nproc=16
#p freq pbepbe/6-311++G(3df,3pd) 

G3 HOF calculation for No.21

 0,1
  O                                                 0.000000000000      0.000000000000      0.000000000000
  C                                                 0.000000000000      0.000000000000      1.372860580000
  C                                                 1.220171316928      0.000000000000     -0.629228201422
  C                                                -1.139845561372      0.000000000000      2.063292149202
  C                                                 1.311383655883      0.000000000000     -1.958748402379
  H                                                 0.982322632826      0.000000000000      1.842291900036
  H                                                 2.087624103317      0.000000000000      0.028684545512
  H                                                -1.110931846348      0.000000000000      3.144356000813
  H                                                 2.285463943339      0.000000000000     -2.428538406281
  H                                                -2.100867568169      0.000000000000      1.564587518938
  H                                                 0.427676124320      0.000000000000     -2.584312631260

 1 2 1.500 3 1.500
 2 1 1.500 4 2.000 6 1.000
 3 1 1.500 5 2.000 7 1.000
 4 2 2.000 8 1.000 10 1.000
 5 3 2.000 9 1.000 11 1.000
 6 2 1.000
 7 3 1.000
 8 4 1.000
 9 5 1.000
 10 4 1.000
 11 5 1.000
 
