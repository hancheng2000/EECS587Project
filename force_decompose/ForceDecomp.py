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

def separate_atoms_init(info, comm):
    rank = comm.Get_rank()    
    # Processor matrix p*p*1
    p = np.sqrt(comm.world)
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
      return atoms_x, atoms_y
    else:
      atoms_x = comm.recv(source=0,tag=0)
      atoms_y = comm.recv(source=0,tag=1)
      atoms = np.concatenate((atoms_x, atoms_y),0)
      return atoms_x, atoms_y
    # return atoms_x, atoms_y, atoms

def gather_force(atoms_x, atoms_y, force, comm,nx_per_processor,ny_per_processor,my_rank=None):
  # Gather all the points into sqrt(comm.world) processors for position update
  p = int(np.sqrt(comm.world))
  x = math.floor(rank / p)
  y = math.floor(rank % p)
  # send the force to transposed matrix for gathering force
  if y>x:
    dest_rank = y * p + x
    comm.send(force,dest=dest_rank, tag=2)
  elif y<x:
    source_rank = y * p + x
    force_other = comm.recv(source = source_rank, tag=2)
    force = np.vstack((force,force_other))
  elif y==x:
    assert force.shape[0] == force.shape[1]
  comm.barrier()

  # send the updated force matrix back to the transposed matrix and apply Newton third law
  if y<x:
    force_send = np.transpose(force) #Newton third law
    dest_rank = y * p + x
    comm.send(force_send, dest = dest_rank, tag = 3)
  elif y>x:
    source_rank = y * p + x
    force_new = comm.recv(source = source_rank, tag = 3)
    force = force_new
  elif y==x:
    assert force.shape[0] == force.shape[1]
  comm.barrier()
  # gather the forces to the first processor in each row
  if y==0:
    subset_ranks = x * p + np.arange(0,p,1)
    subset_comm_group = comm.Create_group(subset_ranks)
    subset_comm = comm.Create(subset_comm_group)
    force_all = np.zeros((force.shape[0],force.shape[1]))
    assert force_all.shape[0] == force_all.shape[1]
    subset_comm.reduce(force,force_all,op=MPI.SUM, root=0)
    # sum up the force to be in the shape (natoms_per_processor, 1)
    final_force = np.sum(force_all,axis=1)
    return final_force
  else:
    return None

#@numba.njit()
def LJ_energy_force(atoms_x, atoms_y,r_cut,L):
  # calculate forces in force decomp
  # For instance, if a processor gets atoms_x=[4,5,6,7], atoms_y=[0,1]
  # then the output force matrix should be 4*2
  # with force[0,1] being f(5,0)
  positions_x = atoms_x[:,0:3]
  positions_y = atoms_y[:,0:3]
  dU_drcut=48*r_cut**(-13)-24*r_cut**(-7)
  dU_drcut_e=24*r_cut**(-7)-48*r_cut**(-13)
  U_rcut_e=4*(r_cut**(-12)-r_cut**(-6))
  force = np.zeros((len(position_x),len(position_y)))
  energy = 0
  for i,px in enumerate(positions_x):
    for j,py in enumerate(positions_y):
      separation = pbc2(separation=px-py,L=L)
      r_relat=np.sqrt(np.sum(separation**2),axis=1)
      #get out the particles inside the r_cut
      if r_relat <= r_cut and r_relat!=0:
        vector_part=separation*(1/r_relat)
        scalar_part=48*r0**(-13)-24*r0**(-7)-dU_drcut
        fc=vector_part*scalar_part
        force[i,j] = fc
        # potential energy
        LJ_num=4*r0**(-12)-4*r0**(-6)-U_rcut-(r0-r_cut)*dU_drcut_e
        energy += LJ_num
  return force, energy

# @numba.njit
# def LJ_potent_nondimen(atoms_x, atoms_y,r_cut,L):
#     ##This function compute the nondimensional potential energy of the system at the given position
#     #Input: position -- the position of all the particles in the sub-system at this instance
#     #       r_cut -- the cutoff for the short-range force field
#     #       L --  the size of the simulation cell
#     #Output: np.sum(update_LJ) -- the potential energy of the sub-system at this instance
#   positions_x = atoms_x[:,0:3]
#   positions_y = atoms_y[:,0:3]
#   num=position.shape[0]
#   update_LJ=np.zeros((num-1,1))
#   #fix value for a certain r_limit
#   dU_drcut=24*r_cut**(-7)-48*r_cut**(-13)
#   U_rcut=4*(r_cut**(-12)-r_cut**(-6))
#   for i,px in enumerate(positions_x):
#     for j, py in enumerate(positions_y):
#       separation = pbc2(separation=px-py,L=L)
#       r_relat=np.sqrt(np.sum(separation**2),axis=1)
#       #get out the particles inside the r_cut
#       if r_relat <= r_cut and r_relat!=0:
#         vector_part=separation*(1/r_relat)
#         scalar_part=48*r0**(-13)-24*r0**(-7)-dU_drcut
#         fc=vector_part*scalar_part
#         force[i,j] = fc        
#     for atom in range(num-1):
#         position_relevent=position[atom:,:]
#         position_other=position_relevent[1:,:]
#         #pbc rule2
#         separation=position_relevent[0,:]-position_other
#         separation_new=pbc2(separation=separation,L=L)
#         r_relat=np.sqrt(np.sum(separation_new**2,axis=1)).reshape(separation_new.shape[0],)
#         LJ=[]
#         #get out the particles inside the r_limit
#         for r0 in r_relat:
#             if r0 <= r_cut:
#                LJ_num=4*r0**(-12)-4*r0**(-6)-U_rcut-(r0-r_cut)*dU_drcut
#                LJ.append(LJ_num)
#             update_LJ[atom,:]=np.sum(np.array(LJ),axis=0)    
#     return np.sum(update_LJ)    
