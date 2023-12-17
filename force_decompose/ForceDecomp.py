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

# We need to think about whether or not we actually need cell_to_dict_force and cell_to_obj_force
# The info list should contain all the information already
def cell_to_dict_force(info,nx,ny,nz,L):
    ##This is the helper function to create dictionary containing matrix for cell_to_obj
    #Input: info -- the position + velocity + acceleration of the patricles

    # In force decomposition essentially we only care about the atom ID
    cell_dict = Dict.empty(key_type=types.int64, value_type=float_array)
    for i in range(1,nx*ny*nz+1): 
      cell_lists[i]= np.zeros((1,9))
    for i in range(info.shape[0]):
      atom=info[i,0:9].reshape(1,9)
    #   The ith row of info represent the ith atom, and atomID is i
      atomID = i
      cell_dict[atomID]=np.append(cell_dict[atomID],atom,axis=0)
    for i in range(1,nx*ny*nz+1):
       cell_dict[i]=cell_dict[i][1:,:]
    return cell_dict

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

def separate_points(info):
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
      atoms_x = info[0:nx_per_processor,:]
      atoms_y = info[0:ny_per_processor,:]
      atoms = np.concatenate((atoms_x,atoms_y),0)
      # TODO: check if using for loop+sendrecv is actually faster than doing bcast. Probably python for too slow
        for i_rank in range(1,world):
            x = math.floor(i_rank / p)
            y = math.floor(i_rank % p)            
            atoms_to_send_x_range = [i_rank * nx_per_processor, (i_rank+1) * nx_per_processor]
            atoms_to_send_y_range = [i_rank * ny_per_processor, (i_rank+1) * ny_per_processor]
            # sending the sub-matrix of atoms to the working processor
            comm.send(info[atoms_to_send_x_range[0]:atoms_to_send_x_range[1]],dest=i_rank,tag=0)
            # we are making use of Newton third law here to only calculate half the interactions.
            # See https://ieeexplore.ieee.org/document/380452 fig.1 for further information
            if x<y:
              comm.send(info[atoms_to_send_y_range[0]:math.floor((atoms_to_send_y_range[0]+atoms_to_send_y_range[1])/2),:],dest=i_rank,tag=1)
            elif x>y:
              comm.send(info[math.floor((atoms_to_send_y_range[0]+atoms_to_send_y_range[1])/2):atoms_to_send_y_range[1],:],dest=i_rank,tag=1)
            else:
              comm.send(info[atoms_to_send_y_range[0]:atoms_to_send_y_range[1],:],dest=i_rank,tag=1)
    else:
      atoms_x = comm.recv(source=0,tag=0)
      atoms_y = comm.recv(source=0,tag=1)
      atoms = np.concatenate((atoms_x, atoms_y),0)
    
    return None

def gather_points(info,my_rank=None):

  pass

#@numba.njit()
def LJ_force(position,neighb_x_0,r_cut,L):
    subcube_atoms=position.shape[0]
    #careful kind of confusing
    position=np.concatenate((position,neighb_x_0),0)
    num=position.shape[0]
    update_accel=np.zeros((subcube_atoms,3))
    dU_drcut=48*r_cut**(-13)-24*r_cut**(-7) 
    #for loop is only of subcube we are interested in but we have to account for ALL distance!
    for atom in range(subcube_atoms):
        position_other=np.concatenate((position[0:atom,:],position[atom+1:num+1,:]),axis=0)
        position_atom=position[atom]
        separation=position_atom-position_other   
        separation_new=pbc2(separation=separation,L=L)
        r_relat=np.sqrt(np.sum(separation_new**2,axis=1))
        #get out the particles inside the r_cut
        accel=np.zeros((r_relat.shape[0],3))
        for i, r0 in enumerate(r_relat):
            if r0 <= r_cut:
               separation_active_num=separation_new[i,:]
               vector_part=separation_active_num*(1/r0)
               scalar_part=48*r0**(-13)-24*r0**(-7)-dU_drcut
               accel_num=vector_part*scalar_part
               accel[i,:]=accel_num
        update_accel[atom,:]=np.sum(accel,axis=0)
    return update_accel.reshape(subcube_atoms,3)

@numba.njit
def LJ_potent_nondimen(position,r_cut,L):
    ##This function compute the nondimensional potential energy of the system at the given position
    #Input: position -- the position of all the particles in the sub-system at this instance
    #       r_cut -- the cutoff for the short-range force field
    #       L --  the size of the simulation cell
    #Output: np.sum(update_LJ) -- the potential energy of the sub-system at this instance
    num=position.shape[0]
    update_LJ=np.zeros((num-1,1))
    #fix value for a certain r_limit
    dU_drcut=24*r_cut**(-7)-48*r_cut**(-13)
    U_rcut=4*(r_cut**(-12)-r_cut**(-6))
    for atom in range(num-1):
        position_relevent=position[atom:,:]
        position_other=position_relevent[1:,:]
        #pbc rule2
        separation=position_relevent[0,:]-position_other
        separation_new=pbc2(separation=separation,L=L)
        r_relat=np.sqrt(np.sum(separation_new**2,axis=1)).reshape(separation_new.shape[0],)
        LJ=[]
        #get out the particles inside the r_limit
        for r0 in r_relat:
            if r0 <= r_cut:
               LJ_num=4*r0**(-12)-4*r0**(-6)-U_rcut-(r0-r_cut)*dU_drcut
               LJ.append(LJ_num)
            update_LJ[atom,:]=np.sum(np.array(LJ),axis=0)    
    return np.sum(update_LJ)    
