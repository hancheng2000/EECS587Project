import numpy as np
import time
import copy
from LJ_SpatialDecomp import *
import utils_spatial_decompose as ut
from mpi4py import MPI

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

#initialize PE, KE, T_insta, P_insta, Momentum
PE=np.zeros((stop_step+1,1))
KE=np.zeros((stop_step+1,1))
T_insta=np.zeros((stop_step+1,1))
P_insta=np.zeros((stop_step+1,1))  

if rank==0:
    start_time = time.time()
    # # initialize
    k_B=1.38064852*10**(-23)
    size_sim=info_init.shape[0]
    # x_dot_init=ut.random_vel_generator(size_sim,T_equal,energy_scale)    
    #initialize the info matrix
    info=np.zeros((stop_step+1,size_sim,9))
    # info[0,:,:]=np.concatenate((position_init,x_dot_init,a_init),axis=1)
    info[0,:,:] = info_init
    # np.savetxt('../data/init.txt',info[0,:,:],fmt='%.6f')
    infotodic=ut.cell_to_obj((info[0,:,:]),subdiv[0],subdiv[1],subdiv[2],L0)
    KE[0,:] = ut.Kin_Eng(info[0,:,3:6])
    T_insta[0,:]=2*KE[0,:]*energy_scale/(3*(size_sim-1)*k_B) #k
    print(T_insta[0,:][0])
else:
    infotodic = None
comm.barrier()
infotodic = comm.bcast(infotodic, root = 0)
T_single = comm.bcast(T_insta[0,:][0], root=0)

my_spd, neighs_spd = ut.separate_points(infotodic, my_rank = rank, nproc = size)
#zero step value
PE_single=ut.LJ_potent_nondimen(my_spd.P,neighs_spd.P,r_cut=r_c,L=L0)
P_single=ut.insta_pressure(L0,T_single,my_spd.P,neighs_spd.P,r_c,energy_scale) #unitless      
PE_all = comm.reduce(PE_single,op=MPI.SUM,root=0)
P_all = comm.reduce(P_single,op=MPI.SUM,root=0)

if rank==0:
    PE[0,:] = PE_all
    P_insta[0,:] = P_all

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
    # apply periodic boundary condition
    my_spd_send[1].P = ut.pbc1(my_spd_send[1].P,L=L0)            
    #calculate and store KE in parallel        
    KE_single = ut.Kin_Eng(my_spd_send[1].V)

    # gather
    temp_infodict=comm.gather(my_spd_send,root=0)
    KE_all = comm.reduce(KE_single,op=MPI.SUM,root=0)
    # T_all = comm.reduce(T_single,op=MPI.SUM,root=0)
    # P_all = comm.reduce(P_single, op=MPI.SUM, root=0)

    if rank == 0:
        if step%20==0:
            print('current time step is ', step)
        temp_infodict=list(filter(None, temp_infodict))
        info_temp=dict(temp_infodict)    
        tmp=ut.concatDict(info_temp)
        info[step+1,:,:]=np.concatenate((tmp.P,tmp.V,tmp.A),1)
        # info[step+1,:,:] = np.round(info[step+1,:,:],4)
        # info[step+1,:,0:3] = ut.pbc1(info[step+1,:,0:3],L=L0)
        #UPDATE CUBES MAKE SURE ATOMS ARE IN RIGHT CUBES
        infotodic=ut.cell_to_obj(info[step+1,:,:],subdiv[0],subdiv[1],subdiv[2],L0)
        # gather KE and T
        KE[step+1,:]=KE_all
        T_insta[step+1,:]=2*KE_all*energy_scale/(3*(size_sim-1)*k_B) #k
    comm.barrier()
    infotodic = comm.bcast(infotodic, root = 0)
    T_single = comm.bcast(T_insta[step+1,:][0], root = 0)
    # calculate PE,P in parallel
    my_spd, neighs_spd = ut.separate_points(infotodic, my_rank = rank, nproc = size)
    PE_single = ut.LJ_potent_nondimen(position = my_spd.P, position_neighbor = neighs_spd.P, r_cut = r_c, L=L0)
    P_single = ut.insta_pressure(L0,T_single,my_spd.P,neighs_spd.P,r_c,energy_scale)
    # gather back to 0
    PE_all = comm.reduce(PE_single,op=MPI.SUM,root=0)
    P_all = comm.reduce(P_single,op=MPI.SUM,root=0)
    if rank==0:
        #calculate and store PE, P_insta
        PE[step+1,:]=PE_all
        P_insta[step+1,:]=P_all
    # comm.barrier()
    # infotodic=comm.bcast(infotodic,root=0)

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
