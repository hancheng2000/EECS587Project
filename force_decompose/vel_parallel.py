from mpi4py import MPI
import numba
import utils_force_decompose as ut
import numpy as np
import functools
import operator
#@numba.njit() ##This might not be needed
def par_worker_vel_ver_pre(atoms_x,dt,r_limit,L,my_rank_y):
    vel_Ver(atoms_x,dt,r_limit,L)
    
#@numba.njit()
def vel_Ver(atoms_x,force,dt,r_limit=2.5,L=6.8):
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    position_old = atoms_x[:,0:3]
    velocity_old = atoms_x[:,3:6]
    accel_old = atoms_x[:,6:9]
    accel_new = force #need to be careful, the original code seems to assume m=1
    velocity_new = velocity_old + accel_new * (dt/2)
    position_new = position_old + velocity_new * dt
    position_new = ut.pbc1(position = position_new,L=L)
    # update postions, velocities and accelerations
    atoms_x[:,0:3] = position_new
    atoms_x[:,3:6] = velocity_new
    atoms_x[:,6:9] = accel_new

    # if rank!=0:
    #   ## Get the data for the current spatial domain 
    #   my_spd, neighs_spd=ut.separate_points(infodict, rank)
    #   #acceleration at 0
    #   my_spd.A=ut.LJ_accel(position=my_spd.P,neighb_x_0=neighs_spd.P,r_cut=r_limit,L=L)
    #   #velocity at dt/2
    #   my_spd.V=my_spd.V+my_spd.A*(dt/2)
    #   #position at dt
    #   my_spd.P=my_spd.P+my_spd.V*dt
    #   #PBC rule 1
    #   my_spd.P=ut.pbc1(position=my_spd.P,L=L)
    #   my_spd_send=(rank,my_spd)
    # else:
    #   my_spd_send=None 
    # comm.barrier()
    # temp_infodict=comm.allgather(my_spd_send)
    # if rank!=0:
    #   temp_infodict=list(filter(None, temp_infodict))
    #   infodict=dict(temp_infodict)
    #   my_spd, neighs_spd=ut.separate_points(infodict, rank) 
    #   #acceleration at dt
    #   my_spd.A=ut.LJ_accel(position=my_spd.P,neighb_x_0=neighs_spd.P,r_cut=r_limit,L=L)
    #   #velocity at dt
    #   my_spd.V=my_spd.V+my_spd.A*(dt/2)
    #   my_spd_send=(rank,my_spd)
    # else:
    #   my_spd_send=None 
    # comm.barrier()
    # temp_infodict=comm.gather(my_spd_send,root=0)
    # if rank==0:
    #   temp_infodict=list(filter(None, temp_infodict))
    #   infodict=dict(temp_infodict)
    return atoms_x
