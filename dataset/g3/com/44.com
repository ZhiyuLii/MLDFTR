%chk=44.chk
%nproc=16
#p freq pbepbe/6-311++G(3df,3pd) 

G3 HOF calculation for No.44

 0,1
  H                                                 1.024400661200     -1.818122831200     -0.616162424300
  C                                                 1.111782155800     -1.454395608700      0.408789534500
  C                                                 0.356605965700     -0.154120916100      0.575476097900
  H                                                 0.522115286400      0.272990022500      1.575029638800
  H                                                 0.716877369100     -2.205412946800      1.096468604500
  H                                                 2.169971238000     -1.292114925600      0.622156052200
  O                                                -1.050854040800     -0.293003505300      0.517204568300
  O                                                 0.845706149700      0.727102674800     -0.409617668200
  C                                                -1.531261615600     -0.867798280100     -0.696464742400
  C                                                 0.330263823900      2.046822271500     -0.260755374100
  H                                                -2.602270886000     -0.665075085200     -0.720363759300
  H                                                 0.831234044100      2.657504440900     -1.011849562700
  H                                                -1.375000532300     -1.951613275100     -0.721165306300
  H                                                -1.052535273300     -0.411509139900     -1.567447555100
  H                                                 0.553185900300      2.441843307000      0.738576973800
  H                                                -0.751136657600      2.075672277800     -0.418210957400

 1 2 1.000
 2 1 1.000 3 1.000 5 1.000 6 1.000
 3 2 1.000 4 1.000 7 1.000 8 1.000
 4 3 1.000
 5 2 1.000
 6 2 1.000
 7 3 1.000 9 1.000
 8 3 1.000 10 1.000
 9 7 1.000 11 1.000 13 1.000 14 1.000
 10 8 1.000 12 1.000 15 1.000 16 1.000
 11 9 1.000
 12 10 1.000
 13 9 1.000
 14 9 1.000
 15 10 1.000
 16 10 1.000
 
