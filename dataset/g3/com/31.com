%chk=31.chk
%nproc=16
#p freq pbepbe/6-311++G(3df,3pd) 

G3 HOF calculation for No.31

 0,1
  H                                                 0.119824933500     -0.879823107200      2.984752554900
  C                                                -0.208879541500      0.000000000000      2.421946669000
  H                                                -1.300486441300      0.000000000000      2.385304001400
  C                                                 0.380084902500      0.000000000000      1.050362621200
  H                                                 0.119824933500      0.879823107200      2.984752554900
  C                                                -0.321784496100      0.000000000000     -0.095091536400
  H                                                 1.469819123700      0.000000000000      0.986970174700
  H                                                -1.409473419800      0.000000000000     -0.106612154400
  C                                                 0.360972393300      0.000000000000     -1.392688814600
  O                                                -0.216590492300      0.000000000000     -2.475245606000
  H                                                 1.470855259300      0.000000000000     -1.340375918500

 1 2 1.000
 2 1 1.000 3 1.000 4 1.000 5 1.000
 3 2 1.000
 4 2 1.000 6 2.000 7 1.000
 5 2 1.000
 6 4 2.000 8 1.000 9 1.000
 7 4 1.000
 8 6 1.000
 9 6 1.000 10 2.000 11 1.000
 10 9 2.000
 11 9 1.000
 
