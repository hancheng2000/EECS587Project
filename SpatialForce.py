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

def cell_to_dict_spatial(info,nx,ny,nz,L):
    ##This is the helper function to create dictionary containing matrix for cell_to_obj
    #Inpust: info -- the position + velocity + acceleration of the patricles at the
    cell_lists = Dict.empty(key_type=types.int64, value_type=float_array)
    xinterval=L/nx
    yinterval=L/ny
    zinterval=L/nz
    #cell_lists={}
    for i in range(1,nx*ny*nz+1): 
      cell_lists[i]= np.zeros((1,9))
    for i in range(info.shape[0]):
      atom=info[i,0:9].reshape(1,9)
      #check extra one!!!
      #if statements !!!!!!!
      #check later
      atomID=int(((np.floor(atom[:,0]/xinterval)+1+(np.floor(atom[:,1]/yinterval))*ny)+(np.floor(atom[:,2]/zinterval))*(nx*ny))[0])
      cell_lists[atomID]=np.append(cell_lists[atomID],atom,axis=0)
    for i in range(1,nx*ny*nz+1):
       cell_lists[i]=cell_lists[i][1:,:]
    return cell_lists

def cell_to_obj_spatial(positions,nx,ny,nz,L):
    cell_lists=cell_to_dict(positions,nx,ny,nz,L)
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

def separate_points_spatial(infodict, my_rank, world):
    neighbor_spd = None
    # Here we need to be careful about only copying the neighboring subcube
    x,y,z = my_rank / world, my_rank % world, my_rank / world / world
    x,y,z = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x_mesh,y_mesh,z_mesh = np.meshgrid(np.linspace(x-1,x+1,3),np.linspace(y-1,y+1,3),np.linspace(z-1,z+1,3))
    neighbor_xyz = np.column_stack((x_mesh.ravel(),y_mesh.ravel(),z_mesh.ravel()))
    # neighbor_rank = np.array([[i-5,i-4,i-3],[i-1,i,i+1],[i+3,i+4,i+5]])
    # for row in neighbor_rank:
    #     for col in row:
    #         if np.floor(col / world) != np.floor(row[1] / world):
    #             col = np.floor(col / world) * world + world
    neighbor_rank = np.zeros(27)
    for t,xyz in enumerate(neighb_xyz):
        for i in xyz:
            if i<0:
                i = i + world
            elif i == world:
                i = i - world
        neighbor_rank[t] = (xyz[0] + xyz[1]*world) + xyz[2]*world*world

    # copy the info in neighboring ranks
    for i, spd in infodict.items():
        if i!=my_rank and i in neighbor_rank:
            if neighb_spd is None:
                neighb_spd = copy.deepcopy(spd)
            else:
                neighb_spd.concat(spd)
    return infodict[my_rank], neighb_spd
