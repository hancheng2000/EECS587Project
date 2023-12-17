import numpy as np
import numba
from numba.experimental import jitclass
from numba import float64
from numba.typed import Dict
from numba.core import types
import os
import pandas as pd
import copy
import math
from mpi4py import MPI

def cell_to_dict_force(info,nx,ny,nz,L):
    ##This is the helper function to create dictionary containing matrix for cell_to_obj
    #Input: info -- the position + velocity + acceleration of the patricles

    # In force decomposition essentially we only care about the atom ID
    cell_dict = Dict.empty(key_type=types.int64, value_type=float_array)
    for i in range(1,nx*ny*nz+1): 
      cell_lists[i]= np.zeros((1,9))
    for i in range(info.shape[0]):
      atom=info[i,0:9].reshape(1,9)
      #check extra one!!!
      #if statements !!!!!!!
      #check later

    #   The ith row of info represent the ith atom, and atomID is i
      atomID = i
      cell_dict[atomID]=np.append(cell_lists[atomID],atom,axis=0)
    for i in range(1,nx*ny*nz+1):
       cell_lists[i]=cell_lists[i][1:,:]
    return cell_lists

def cell_to_obj_force(positions,nx,ny,nz,L):
    cell_lists=cell_to_dict_force(positions,nx,ny,nz,L)
    new_cell_list={}
    for i in range(1,nx*ny*nz+1):
      # I think this works but we should think more about what an "empty" shape means
      # currently when a key points to an empty cube the cube has a position vector of dimensions [0,3]
      if cell_lists[i].shape[0]!=0:
        temp_reshaped=cell_lists[i]
        temp_position = temp_reshaped[:,0:3]
        assert(temp_position.shape[1] == 3)
        temp_velocity = temp_reshaped[:,3:6]
        temp_acceleration = temp_reshaped[:,6:9]
        spatial_domain_data = SpatialDomainData(temp_position,
                                                temp_velocity,
                                                temp_acceleration)
        new_cell_list[i] = spatial_domain_data
      else:
        new_cell_list[i] = SpatialDomainData(np.empty((0,3)),
                                          np.empty((0,3)),
                                          np.empty((0,3)))
    return new_cell_list

def separate_points_force(info):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()    
    # Processor matrix p*p*1
    p = np.sqrt(world)
    x = math.floor(rank / p)
    y = math.floor(rank % p)
    natoms = len(info)
    nx_per_processor = natoms / p
    ny_per_processor = natoms / p

    # comm.bcast(info,root=0)
    if rank == 0:
        for i_rank in range(world):
            x = math.floor(i_rank / p)
            y = math.floor(i_rank % p)            
            atoms_to_send_x_range = [i_rank * nx_per_processor, (i_rank+1) * nx_per_processor]
            atoms_to_send_y_range = [i_rank * ny_per_processor, (i_rank+1) * ny_per_processor]
            # sending the sub-matrix of atoms to the working processor
            comm.send(info[atoms_to_send_x_range[0]:atoms_to_send_x_range[1]],dest=i_rank,tag=0)
            comm.send(info[atoms_to_send_y_range[0]:atoms_to_send_y_range[1]],dest=i_rank,tag=1)
    else:
      atoms_x = comm.recv(source=0,tag=0)
      atoms_y = comm.recv(source=0,tag=1)
      atoms = np.concatenate((atoms_x, atoms_y),0)
    
    return None
