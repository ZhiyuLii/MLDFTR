%chk=45.chk
%nproc=16
#p freq pbepbe/6-311++G(3df,3pd) 

G3 HOF calculation for No.45

 0,1
  C                                                 0.005734474000      0.000000000000     -0.341185138700
  S                                                 0.030400075700      0.000000000000      1.495083796800
  C                                                 1.470942503400      0.000000000000     -0.769320501000
  C                                                -0.693829878800     -1.252748781400     -0.857905806400
  C                                                -0.693829878800      1.252748781400     -0.857905806400
  H                                                -1.302684129400      0.000000000000      1.659874970700
  H                                                 1.530735304000      0.000000000000     -1.862296443800
  H                                                 1.990371602100     -0.887748555800     -0.397554797300
  H                                                 1.990371602100      0.887748555800     -0.397554797300
  H                                                -0.669806407400     -1.267054470700     -1.953488294500
  H                                                -1.741787490100     -1.278085344800     -0.543387831400
  H                                                -0.203055557000     -2.155372778400     -0.486076957300
  H                                                -0.669806407400      1.267054470700     -1.953488294500
  H                                                -1.741787490100      1.278085344800     -0.543387831400
  H                                                -0.203055557000      2.155372778400     -0.486076957300

 1 2 1.000 3 1.000 4 1.000 5 1.000
 2 1 1.000 6 1.000
 3 1 1.000 7 1.000 8 1.000 9 1.000
 4 1 1.000 10 1.000 11 1.000 12 1.000
 5 1 1.000 13 1.000 14 1.000 15 1.000
 6 2 1.000
 7 3 1.000
 8 3 1.000
 9 3 1.000
 10 4 1.000
 11 4 1.000
 12 4 1.000
 13 5 1.000
 14 5 1.000
 15 5 1.000
 
