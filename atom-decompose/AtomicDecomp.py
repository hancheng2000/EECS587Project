import numpy as np
import numba
from numba.experimental import jitclass
from numba import float64
from numba.typed import Dict
from numba.core import types
import os
import pandas as pd
import copy
from utils import SpatialDomainData as spd

def cell_to_dict_atom(info,ncore):
    cell_lists = Dict.empty(key_type=types.int64, value_type=float_array)
    for i in range(1,ncore+1):
        cell_lists[i] = np.zeros((1,9))
    for i in range(info.shape[0]):
        # atomic coordinates, velocity and acceleration
        atom = info[i,0:9].reshape(1,9)
        atomID = int(np.floor(i*ncore/info.shape[0])) + 1
        cell_lists[atomID] = np.append(cell_lists[atomID],atom,axis=0)
    # for i in range(1,ncore):
    #     cell_lists[i] = cell_lists[i][1:,:]
    return cell_lists

def cell_to_obj_atom(info,ncore):
    cell_lists = cell_to_dict_atom(info,ncore)
    new_cell_list = {}
    for i in range(1, ncore+1):
        if cell_lists[i].shape[0]!=0:
            pos = cell_lists[i][:,0:3] # position
            vel = cell_lists[i][:,3:6] # velocity
            acc = cell_lists[i][:,6:9] # acceleration
            spatial_domain_data = spd(pos, vel, acc)
            new_cell_list[i] = spatial_domain_data
        else:
            new_cell_list[i] = spd(np.empty((0,3)),np.empty((0,3)),np.empty((0,3)))
    return new_cell_list

def separate_points_atom(infodict,my_rank):
    # since we are using atom-wise decomposition, each processor must know every atomic position
    neighbor_spd = None
    for i, spd in infodict.items():
        if i!=my_rank:
            if neighbor_spd is None:
                neighbor_spd = copy.deepcopy(spd)
            else:
                neighbor_spd.concat(spd)
    return infodict[my_rank], neighbor_spd

