import numpy as np
import time
import copy
from LJ_ForceDecomp import *
import utils_force_decompose as ut
from mpi4py import MPI
stop_step=1
k_B=1.38064852*10**(-23)
dt=0.002
file_name='initial_position_LJ_argon256.txt'
position_init=np.loadtxt(file_name)
num_atoms=position_init.shape[0]

a_init=np.zeros((num_atoms,3))
r_c=2.5
L0=6.8
#cube division
subdiv=np.array([2,2,1])
energy_scale=1.66*10**(-21)#unit: J, Argon
sigma=3.4 #unit: Ang, Argon 
T_dimensional_equal=300 #unit K
T_equal=T_dimensional_equal*k_B/energy_scale
part_type='LJ'
name='argon256'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank==0:
    start_time=time.time()
    info,PE,KE,T_insta,P_insta=LJ_MD(subdiv=subdiv,
                                    position_init=position_init,
                                    dt=dt,
                                    stop_step=stop_step,
                                    accel_init=a_init,
                                    r_cut=2.8, 
                                    L=L0,
                                    T_eq=T_equal,
                                    e_scale=energy_scale,
                                    sig=sigma)
    end_time=time.time()
    period = end_time-start_time
    print("For this {} steps operation (dt={}) with r_cut = {} and L = {}, it took:".format(stop_step,dt,r_c,L0),end_time-start_time,'s')
    ut.data_saver(info, PE, KE, T_insta, P_insta, L0, num_atoms,part_type,name,period, stop_step, r_c, 1000, False)
else:
    LJ_MD(subdiv=subdiv,
        position_init=position_init,
        dt=dt,
        stop_step=stop_step,
        accel_init=a_init,
        r_cut=2.8, 
        L=L0,
        T_eq=T_equal,
        e_scale=energy_scale,
        sig=sigma)
