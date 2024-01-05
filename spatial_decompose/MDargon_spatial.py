import numpy as np
import time
import copy
from LJ_SpatialDecomp import *
# import utils_spatial_decompose as ut
from utils_spatial_decompose import data_saver
from mpi4py import MPI

class SpatialDomainData:
    #This class creates objects for storing position, velocity and acceleration information
    ##This class contain three functions:
    ###__init__ (description)
    ###__repr__ (description)
    ###concat(description)
    def __init__(self, position=np.array([]), velocity=np.array([]), acceleration=np.array([])):
        self.P = position
        self.V = velocity
        self.A = acceleration

    def __repr__(self):
        return "(P:{}, V:{}, A:{})".format(self.P, self.V, self.A)

    def concat(self, other):
        self.P = np.concatenate((self.P, other.P), 0)
        self.V = np.concatenate((self.V, other.V), 0)
        self.A = np.concatenate((self.A, other.A), 0)

def LJ_acc(position,neighb_x_0,r_cut,L, rank):
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
        #     print(accel[accel.any(axis=1)],r_relat[accel.any(axis=1)])
    return update_accel.reshape(subcube_atoms,3)

def pbc1(position,L):
    ##This function perform the first rule of periodic boundary condition 
    #Input: position -- the position before PBC1
    #       L        -- the simulation cell size
    #Ouput: x_new    -- the updated position after PBC1
    position_new=np.zeros((position.shape[0],3))   
    for i in range(position.shape[0]):
        position_ind=position[i,:]
        position_empty=np.zeros((1,3))
        for j in range(3):
            # position_axis=numba.float64(position_ind[j])
            position_axis = position_ind[j]
            if position_axis < 0:
                position_axis_new=position_axis+L
            elif position_axis >= L:
                position_axis_new=position_axis-L
            else:
                position_axis_new=position_axis
            position_empty[:,j]=position_axis_new
        position_new[i,:]=position_empty
    return position_new

# @numba.njit
def pbc2(separation,L):
    ##This function perform the second rule of periodic boundary condition 
    #Input: separation -- separation before PBC2
    #       L          -- the simulation cell size
    #Ouput: separation_new -- the updated separation after PBC2
    separation_new=np.zeros((separation.shape[0],3))   
    for i in range(separation.shape[0]):
        separation_ind=separation[i,:]
        separation_empty=np.zeros((1,3))
        for j in range(3):
            # separation_axis=numba.float64(separation_ind[j])
            separation_axis = separation_ind[j]
            if separation_axis < -L/2:
                separation_axis_new=separation_axis+L
            elif separation_axis >= L/2:
                separation_axis_new=separation_axis-L
            else:
                separation_axis_new=separation_axis
            separation_empty[:,j]=separation_axis_new
        separation_new[i,:]=separation_empty
    return separation_new    

def separate_points(infodict, my_rank, nproc):
    neighb_spd = None
    # Processor matrix n*n*1 (for comparison with force decomposition)
    axis = np.sqrt(nproc)
    # Here we need to be careful about only copying the neighboring subcube
    x,y,z = my_rank % axis, my_rank / axis, 0
    x,y,z = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x_mesh,y_mesh = np.meshgrid(np.linspace(x-1,x+1,3),np.linspace(y-1,y+1,3))
    neighbor_xy = np.column_stack((x_mesh.ravel(),y_mesh.ravel()))
    # neighbor_rank = np.array([[i-5,i-4,i-3],[i-1,i,i+1],[i+3,i+4,i+5]])
    # for row in neighbor_rank:
    #     for col in row:
    #         if np.floor(col / world) != np.floor(row[1] / world):
    #             col = np.floor(col / world) * world + world
    if axis<=3:
        neighbor_rank = np.arange(0,nproc,1)
    else:
        neighbor_rank = np.zeros(9)
        for t,xy in enumerate(neighbor_xy):
            xy_old = xy.copy()
            if xy[0]<0:
                xy[0] = xy[0] + axis
            if xy[0]==axis:
                xy[0] = xy[0] - axis
            # elif xy[0]==axis and xy[1]<axis-1:
            #     xy[0] = xy[0] - axis
            #     xy[1] = xy[1] + 1
            # elif xy[0]==axis and xy[1]>=axis-1:
            #     xy[0] = xy[0] - axis
            #     xy[1] = 0
            if xy[1]<0:
                xy[1] = xy[1] + axis
            if xy[1] == axis:
                xy[1] = xy[1] - axis
                # xy[0] = xy[0]
            # print(xy_old,xy)
            # for i in xy:
            #     if i<0:
            #         i = i + axis
            #     elif i == axis:
            #         i = i - axis
            neighbor_rank[t] = (xy[0] + xy[1]*axis)
    # print('neighboring ',my_rank, (x,y), neighbor_rank)

    # copy the info in neighboring ranks
    for i, spd in infodict.items():
        if i!=my_rank and i in neighbor_rank:
            if neighb_spd is None:
                neighb_spd = copy.deepcopy(spd)
            else:
                neighb_spd.concat(spd)
    return infodict[my_rank], neighb_spd

def cell_to_dict(info,nx,ny,nz,L):
    ##This is the helper function to create dictionary containing matrix for cell_to_obj
    #Inpust: info -- the position + velocity + acceleration of the patricles at the
    # cell_dict = Dict.empty(key_type=types.int64, value_type=float_array)
    cell_dict = {}
    xinterval=L/nx
    yinterval=L/ny
    zinterval=L/nz
    # print('interval ', xinterval, yinterval)
    #cell_lists={}
    for i in range(nx*ny*nz): 
      cell_dict[i]= np.zeros((1,9))
    for i in range(info.shape[0]):
      atom=info[i,0:9].reshape(1,9)
      #check extra one!!!
      #if statements !!!!!!!
      #check later
    #   atomID=int(((np.floor(atom[:,0]/xinterval)+(np.floor(atom[:,1]/yinterval))*ny)+(np.floor(atom[:,2]/zinterval))*(nx*ny))[0])
      atomID=int(((np.floor(atom[:,0]/xinterval)+(np.floor(atom[:,1]/yinterval))*ny))[0])
      cell_dict[atomID]=np.append(cell_dict[atomID],atom,axis=0)
    for i in range(nx*ny*nz):
       cell_dict[i]=cell_dict[i][1:,:]
    return cell_dict

def cell_to_obj(positions,nx,ny,nz,L):
    cell_lists=cell_to_dict(positions,nx,ny,nz,L)
    new_cell_list={}
    for i in range(nx*ny*nz):
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

def Kin_Eng(velocity):
    ##This function compute the average kinetic energy of the system at given the velocity
    #Input: velocity -- the velocities of all the particles
    #Output: Kinetic_sum -- the average kinetic energy of the system
    num=velocity.shape[0]
    Kinetic_per=0.5*np.sum(velocity**2,axis=1)
    Kinetic_avg=np.sum(Kinetic_per,axis=0)
    return Kinetic_avg

def LJ_potent_nondimen(position,position_neighbor,r_cut,L):
    ##This function compute the nondimensional potential energy of the system at the given position
    #Input: position -- the position of all the particles at this instance
    #       r_cut -- the cutoff for the short-range force field
    #       L --  the size of the simulation cell
    #Output: np.sum(update_LJ) -- the potential energy of the system at this instance
    num=position.shape[0]
    position = np.concatenate((position, position_neighbor),0)
    update_LJ=np.zeros((num,1))
    #fix value for a certain r_limit
    dU_drcut=24*r_cut**(-7)-48*r_cut**(-13)
    U_rcut=4*(r_cut**(-12)-r_cut**(-6))
    for atom in range(num):
        # position_relevent=position[atom:,:]
        # position_other=position_relevent[1:,:]
        position_atom = position[atom,:]
        position_other=np.concatenate((position[0:atom,:],position[atom+1:,:]),axis=0)
        #pbc rule2
        # separation=position_relevent[0,:]-position_other
        separation = position_atom - position_other
        separation_new=pbc2(separation=separation,L=L)
        r_relat=np.sqrt(np.sum(separation_new**2,axis=1)).reshape(separation_new.shape[0],)
        LJ=[]
        #get out the particles inside the r_limit
        for r0 in r_relat:
            if r0 <= r_cut and r0!=0:
               LJ_num=4*r0**(-12)-4*r0**(-6)-U_rcut-(r0-r_cut)*dU_drcut
               LJ.append(LJ_num)
            update_LJ[atom,:]=np.sum(np.array(LJ),axis=0)    
    return np.sum(update_LJ)/2

def insta_pressure(L,T,position,position_neighbor,r_cut,e_scale):
    ##This function computes the dimensionless pressure
    #Inputs: L -- The size of the simulation cell
    #        T -- The simulation temperature
    #        position -- The position of all the particles at this time step
    #        e_scale -- The energy scale of this particles
    #Ouputs: pres_insta -- The instant pressure at this moment
    num=position.shape[0]
    k_B=1.38064852*10**(-23)
    V=L**3
    pres_ideal=num*T*(k_B/e_scale)/V
    dU_drcut=24*r_cut**(-7)-48*r_cut**(-13)
    pres_virial=np.zeros((num,1))
    position = np.concatenate((position,position_neighbor),0)
    for atom in range(num):
        # position_relevent=position[atom:,:]
        # position_other=position_relevent[1:,:]
        # position_atom=position_relevent[0,:]
        position_atom = position[atom,:]
        position_other = np.concatenate((position[0:atom,:],position[atom+1:,:]),axis=0)
        #pbc rule 2
        separation=position_atom-position_other
        separation_new=pbc2(separation=separation,L=L)
        r_relat=np.sqrt(np.sum(separation_new**2,axis=1)).reshape(separation_new.shape[0],)
        force=[]
        active_r_relat=[]
        #get out the particles inside the r_limit
        for r0 in r_relat:
            if r0 <= r_cut and r0!=0:
               #active_r_relat.append(r0)
               force_num=-(24*r0**(-7)-48*r0**(-13))+dU_drcut
               force.append(force_num)
               active_r_relat.append(r0)    
        #get out the particles inside the r_cut
        active_amount=np.array(active_r_relat).shape[0]
        rijFij=np.array(active_r_relat).reshape(1,active_amount)@np.array(force).reshape(active_amount,1)
        pres_virial[atom,:]=rijFij
    pres_insta=pres_ideal+np.sum(pres_virial,axis=0)/(3*V)/2
    return pres_insta

def concatDict(infodict):
  wholeDict=None
  for i,spd in infodict.items():
      if(wholeDict is None):
        wholeDict = copy.deepcopy(spd)
      else:
        wholeDict.concat(spd)
  return wholeDict

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
    infotodic=cell_to_obj((info[0,:,:]),subdiv[0],subdiv[1],subdiv[2],L0)
    KE[0,:] = Kin_Eng(info[0,:,3:6])
    T_insta[0,:]=2*KE[0,:]*energy_scale/(3*(size_sim-1)*k_B) #k
    print(T_insta[0,:][0])
else:
    infotodic = None
comm.barrier()
infotodic = comm.bcast(infotodic, root = 0)
T_single = comm.bcast(T_insta[0,:][0], root=0)

my_spd, neighs_spd = separate_points(infotodic, my_rank = rank, nproc = size)
#zero step value
PE_single=LJ_potent_nondimen(my_spd.P,neighs_spd.P,r_cut=r_c,L=L0)
P_single=insta_pressure(L0,T_single,my_spd.P,neighs_spd.P,r_c,energy_scale) #unitless      
PE_all = comm.reduce(PE_single,op=MPI.SUM,root=0)
P_all = comm.reduce(P_single,op=MPI.SUM,root=0)

if rank==0:
    PE[0,:] = PE_all
    P_insta[0,:] = P_all

# Run MD
for step in range(stop_step):
    my_spd, neighs_spd=separate_points(infotodic, rank, size)
    my_spd.A = LJ_acc(position=my_spd.P,neighb_x_0=neighs_spd.P,r_cut=r_c,L=L0,rank=rank)
    my_spd.V = my_spd.V + my_spd.A*(dt)
    my_spd.P = my_spd.P + my_spd.V * dt
    my_spd.P = pbc1(position=my_spd.P,L=L0)
    # my_spd_send=utp.vel_Ver(
    #     comm = comm,
    #     infodict=infotodic,
    #     dt=dt,
    #     r_limit=r_c,
    #     L=L0,
    #     my_rank=rank
    #     )
    my_spd_send = (rank,my_spd)
    # apply periodic boundary condition
    # my_spd_send[1].P = ut.pbc1(my_spd_send[1].P,L=L0)            
    #calculate and store KE in parallel        
    KE_single = Kin_Eng(my_spd_send[1].V)

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
        tmp = concatDict(info_temp)
        info[step+1,:,:] = np.concatenate((tmp.P,tmp.V,tmp.A),1)
        # info[step+1,:,:] = np.round(info[step+1,:,:],4)
        # info[step+1,:,0:3] = ut.pbc1(info[step+1,:,0:3],L=L0)
        #UPDATE CUBES MAKE SURE ATOMS ARE IN RIGHT CUBES
        infotodic = cell_to_obj(info[step+1,:,:],subdiv[0],subdiv[1],subdiv[2],L0)
        # gather KE and T
        KE[step+1,:]=KE_all
        T_insta[step+1,:]=2*KE_all*energy_scale/(3*(size_sim-1)*k_B) #k
    comm.barrier()
    infotodic = comm.bcast(infotodic, root = 0)
    T_single = comm.bcast(T_insta[step+1,:][0], root = 0)
    # calculate PE,P in parallel
    my_spd, neighs_spd = separate_points(infotodic, my_rank = rank, nproc = size)
    PE_single = LJ_potent_nondimen(position = my_spd.P, position_neighbor = neighs_spd.P, r_cut = r_c, L=L0)
    P_single = insta_pressure(L0,T_single,my_spd.P,neighs_spd.P,r_c,energy_scale)
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
    data_saver(info, PE, KE, T_insta, P_insta, L0, num_atoms,part_type,name,period, stop_step, r_c, size, False)
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
