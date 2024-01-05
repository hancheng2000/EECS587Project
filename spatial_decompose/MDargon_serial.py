import numpy as np
import time
import copy
from LJ_SpatialDecomp import *
# import utils_spatial_decompose as ut
import utils as ut
import utils_spatial_decompose
from mpi4py import MPI


def LJ_acc(position,r_cut,L, rank):
    subcube_atoms=position.shape[0]
    #careful kind of confusing
    num=position.shape[0]
    update_accel=np.zeros((subcube_atoms,3))
    dU_drcut=48*r_cut**(-13)-24*r_cut**(-7) 
    #for loop is only of subcube we are interested in but we have to account for ALL distance!
    for atom in range(subcube_atoms):
        position_other=np.concatenate((position[0:atom,:],position[atom+1:num+1,:]),axis=0)
        position_atom=position[atom]
        separation=position_atom-position_other   
        separation_new=ut.pbc2(separation=separation,L=L)
        r_relat=np.sqrt(np.sum(separation_new**2,axis=1))
        #get out the particles inside the r_cut
        accel=np.zeros((r_relat.shape[0],3))
        flag = 0
        for i, r0 in enumerate(r_relat):
            if r0 <= r_cut and r0!=0:
               separation_active_num=separation_new[i,:]
               vector_part=separation_active_num*(1/r0)
               scalar_part=48*r0**(-13)-24*r0**(-7)-dU_drcut
               accel_num=vector_part*scalar_part
               accel[i,:]=accel_num
            #    if np.abs(position_atom[0]-4.641496)<0.001 and np.abs(position_atom[1]-3.092705)<0.001 and np.abs(position_atom[2]-3.090466)<0.001:
            #        flag = 1
            #        print(position_atom, accel_num, r0)
        update_accel[atom,:]=np.sum(accel,axis=0)
        # if np.abs(update_accel[atom,:][0]+6.2848242)<1e-4:
        #     print('update accel',update_accel[atom,:])
        #     print(accel[accel.any(axis=1)],r_relat[a
    return update_accel.reshape(subcube_atoms,3)

# # MPI initialize
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# print(rank,size)
# comm.barrier()
rank = 0
size = 1

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

# if rank==0:

# # initialize
k_B=1.38064852*10**(-23)
size_sim=info_init.shape[0]
# x_dot_init=ut.random_vel_generator(size_sim,T_equal,energy_scale)
#initialize the info matrix
info=np.zeros((stop_step+1,size_sim,9))
# info[0,:,:]=np.concatenate((position_init,x_dot_init,a_init),axis=1)
start_time = time.time()
info[0,:,:] = info_init
# np.savetxt('../data/init.txt',info[0,:,:],fmt='%.6f')
# infotodic=ut.cell_to_obj((info[0,:,:]),subdiv[0],subdiv[1],subdiv[2],L0)
my_spd = ut.SpatialDomainData(info[0,:,0:3],info[0,:,3:6],info[0,:,6:9])
#zero step value
PE[0,:]=ut.LJ_potent_nondimen(info[0,:,0:3],r_cut=r_c,L=L0)/2
KE[0,:]=ut.Kin_Eng(info[0,:,3:6])
T_insta[0,:]=2*KE[0,:]*energy_scale/(3*(size_sim-1)*k_B) #k
P_insta[0,:]=ut.insta_pressure(L0,T_insta[0],info[0,:,0:3],r_c,energy_scale)/2 #unitless      

# Run MD
for step in range(stop_step):
    my_spd.A = LJ_acc(position=my_spd.P,r_cut=r_c,L=L0,rank=0)
    my_spd.V=my_spd.V+my_spd.A*(dt)
    my_spd.P=my_spd.P+my_spd.V*dt
    my_spd.P=ut.pbc1(position=my_spd.P,L=L0)
    if step%20==0:
        print('current time step is ', step)
    # temp_infodict = (0,my_spd)
    # temp_infodict=list(filter(None, temp_infodict))
    temp_infodict = [(0,my_spd)]
    info_temp=dict(temp_infodict)
    # print(info_temp)
    tmp=ut.concatDict(info_temp)
    info[step+1,:,:]=np.concatenate((tmp.P,tmp.V,tmp.A),1)
    # info[step+1,:,:] = np.round(info[step+1,:,:],4)
    info[step+1,:,0:3] = ut.pbc1(info[step+1,:,0:3],L=L0)
    #UPDATE CUBES MAKE SURE ATOMS ARE IN RIGHT CUBES
    # infotodic=ut.cell_to_obj(info[step+1,:,:],subdiv[0],subdiv[1],subdiv[2],L0)
    
    #calculate and store PE, KE, T_insta, P_insta in parallel
    PE[step+1,:]=ut.LJ_potent_nondimen(info[step+1,:,0:3],r_cut=r_c,L=L0)
    KE[step+1,:]=ut.Kin_Eng(info[step+1,:,3:6])
    T_insta[step+1,:]=2*KE[step+1,:]*energy_scale/(3*(size_sim-1)*k_B) #k
    P_insta[step+1,:]=ut.insta_pressure(L0,T_insta[step+1],info[step+1,:,0:3],r_c,energy_scale) #unitless


end_time = time.time()
period = end_time-start_time
print(f'Run time is {period:.3f} seconds')
PE = np.round(PE,4)
KE = np.round(KE,4)
T_insta = np.round(T_insta,4)
P_insta = np.round(P_insta,4)
utils_spatial_decompose.data_saver(info, PE, KE, T_insta, P_insta, L0, num_atoms,part_type,name,period, stop_step, r_c, size, False)

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
