from mpi4py import MPI
# import numba
import utils_spatial_decompose as ut
import numpy as np
import functools
import operator
#@numba.njit() ##This might not be needed
def par_worker_vel_ver_pre(comm,infodict,dt,r_limit,L,my_rank):
    vel_Ver(comm,infodict,dt,r_limit,L,my_rank)
    
#@numba.njit()
def vel_Ver(comm,infodict,dt,r_limit=2.5,L=6.8,my_rank=0):
  # rank = comm.Get_rank()
  rank = my_rank
  size = comm.Get_size()
  print(rank,size)
  ## Get the data for the current spatial domain 
  my_spd, neighs_spd=ut.separate_points(infodict, rank, size)
  # LEAP FROG METHOD
  #acceleration at 0
  my_spd.A=ut.LJ_accel(position=my_spd.P,neighb_x_0=neighs_spd.P,r_cut=r_limit,L=L)
  #velocity at dt/2
  my_spd.V=my_spd.V+my_spd.A*(dt/2)
  #position at dt
  my_spd.P=my_spd.P+my_spd.V*dt
  if rank==0:
    print('before pbc1 ',my_spd.P)  
  #PBC rule 1
  my_spd.P=ut.pbc1(position=my_spd.P,L=L)
  my_spd_send=(rank,my_spd)
  comm.barrier()
  temp_infodict=comm.allgather(my_spd_send)
  # acceleration at dt
  my_spd.A=ut.LJ_accel(position=my_spd.P,neighb_x_0=neighs_spd.P,r_cut=r_limit,L=L)
  # velocity at dt
  my_spd.V=my_spd.V+my_spd.A*(dt/2)
  my_spd_send=(rank,my_spd)
  comm.barrier()
  return my_spd_send
