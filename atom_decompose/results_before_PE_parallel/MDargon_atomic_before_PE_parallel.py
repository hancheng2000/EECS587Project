import numpy as np
import time
import copy
from LJ_AtomicDecomp import *
import utils_atomic_decompose as ut
from mpi4py import MPI
import utils


# MPI initialize
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank,size)
comm.barrier()

# run params
stop_step=1000
k_B=1.38064852*10**(-23)
dt=0.002
file_name='../data/initial_position_LJ_argon256.txt'
info_init=np.loadtxt(file_name)
num_atoms=info_init.shape[0]

a_init=np.zeros((num_atoms,3))
r_c=1.6
L0=6.8
#cube division
subdiv=np.array([int(np.sqrt(size)),int(np.sqrt(size)),1])
energy_scale=1.66*10**(-21)#unit: J, Argon
sigma=3.4 #unit: Ang, Argon 
T_dimensional_equal=300#unit K
T_equal=T_dimensional_equal*k_B/energy_scale
part_type='LJ'
name='argon256'


if rank==0:
    start_time = time.time()
    # # initialize
    k_B=1.38064852*10**(-23)
    size_sim=info_init.shape[0]
    # x_dot_init=ut.random_vel_generator(size_sim,T_equal,energy_scale)
    #initialize PE, KE, T_insta, P_insta, Momentum
    PE=np.zeros((stop_step+1,1))
    KE=np.zeros((stop_step+1,1))
    T_insta=np.zeros((stop_step+1,1))
    P_insta=np.zeros((stop_step+1,1))      
    #initialize the info matrix
    info=np.zeros((stop_step+1,size_sim,9))
    # info[0,:,:]=np.concatenate((position_init,x_dot_init,a_init),axis=1)
    info[0,:,:] = info_init
    # np.savetxt('../data/init.txt',info[0,:,:],fmt='%.6f')
    infotodic=ut.cell_to_obj((info[0,:,:]),size)
    #zero step value
    PE[0,:]=utils.LJ_potent_nondimen(info[0,:,0:3],r_cut=r_c,L=L0)
    KE[0,:]=ut.Kin_Eng(info[0,:,3:6])
    T_insta[0,:]=2*KE[0,:]*energy_scale/(3*(size_sim-1)*k_B) #k
    P_insta[0,:]=utils.insta_pressure(L0,T_insta[0],info[0,:,0:3],r_c,energy_scale) #unitless      
else:
    infotodic = None
    # info = None
comm.barrier()
infotodic = comm.bcast(infotodic, root = 0)

# Run MD
for step in range(stop_step):
    my_spd_send=utp.vel_Ver(
        comm = comm,
        infodict=infotodic,
        dt=dt,
        r_limit=r_c,
        L=L0,
        my_rank=rank
        )
    temp_infodict=comm.gather(my_spd_send,root=0)
    if rank == 0:
        if step%20==0:
            print('current time step is ', step)
        temp_infodict=list(filter(None, temp_infodict))
        info_temp=dict(temp_infodict)    
        tmp=ut.concatDict(info_temp)
        info[step+1,:,:]=np.concatenate((tmp.P,tmp.V,tmp.A),1)
        # info[step+1,:,:] = np.round(info[step+1,:,:],4)
        info[step+1,:,0:3] = ut.pbc1(info[step+1,:,0:3],L=L0)
        #UPDATE CUBES MAKE SURE ATOMS ARE IN RIGHT CUBES
        infotodic=ut.cell_to_obj(info[step+1,:,:],size)
        #calculate and store PE, KE, T_insta, P_insta
        PE[step+1,:]=utils.LJ_potent_nondimen(info[step+1,:,0:3],r_cut=r_c,L=L0)
        KE[step+1,:]=ut.Kin_Eng(info[step+1,:,3:6])
        T_insta[step+1,:]=2*KE[step+1,:]*energy_scale/(3*(size_sim-1)*k_B) #k
        P_insta[step+1,:]=utils.insta_pressure(L0,T_insta[step+1],info[step+1,:,0:3],r_c,energy_scale) #unitless
    comm.barrier()
    infotodic=comm.bcast(infotodic,root=0)

if rank == 0:
    end_time = time.time()
    period = end_time-start_time
    print(f'Run time is {period:.3f} seconds')
    PE = np.round(PE,4)
    KE = np.round(KE,4)
    T_insta = np.round(T_insta,4)
    P_insta = np.round(P_insta,4)
    ut.data_saver(info, PE, KE, T_insta, P_insta, L0, num_atoms,part_type,name,period, stop_step, r_c, size, False)
comm.barrier()

# if rank==0:
#     start_time=time.time()
#     info,PE,KE,T_insta,P_insta=LJ_MD(subdiv=subdiv,
#                                     position_init=position_init,
#                                     dt=dt,
#                                     stop_step=stop_step,
#                                     accel_init=a_init,
#                                     r_cut=2.8, 
#                                     L=L0,
#                                     T_eq=T_equal,
#                                     e_scale=energy_scale,
#                                     sig=sigma)
#     end_time=time.time()
#     period = end_time-start_time
#     print("For this {} steps operation (dt={}) with r_cut = {} and L = {}, it took:".format(stop_step,dt,r_c,L0),end_time-start_time,'s')
#     ut.data_saver(info, PE, KE, T_insta, P_insta, L0, num_atoms,part_type,name,period, stop_step, r_c, 1000, False)
# else: