import numpy as np

L0=6.8
a = np.loadtxt('initial_position_LJ_argon256.txt')
b = [a.copy() for i in range(8)]
b[0] = a.copy()
b[1][:,0] = b[1][:,0]+6.8
b[2][:,1] = b[2][:,1]+6.8
b[3][:,0] = b[3][:,0]+6.8
b[3][:,1] = b[3][:,1]+6.8
b[4] = b[0].copy()
b[4][:,2] = b[4][:,2]+6.8
b[5] = b[1].copy()
b[5][:,2] = b[5][:,2]+6.8
b[6] = b[2].copy()
b[6][:,2] = b[6][:,2]+6.8
b[7] = b[3].copy()
b[7][:,2] = b[7][:,2]+6.8

out = np.concatenate((b[0],b[1]),axis=0)
for i in range(2,8):
    out = np.concatenate((out,b[i]),axis=0)
np.savetxt('initial_position_LJ_argon2048.txt',out,fmt='%.5f')