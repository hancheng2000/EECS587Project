import numpy as np
# import numba
# from numba.experimental import jitclass
# from numba import float64
# from numba.typed import Dict
# from numba.core import types
import os
import pandas as pd
import copy
#first the dtypes of the input matrices need to be explicitly clarified
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

# @numba.njit
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
                print(position_axis)
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

# @numba.njit
def random_vel_generator(n,T_equal,e_scale): 
    ##This function generate random initial velocity at the desired temperature 
    #Input: n -- number of atoms
    #       T_equal -- the temperature of interests
    #       e_scale -- the energy scale of the particle of interests
    #Ouput: vel_per_particle -- the velocity matrix assigned to all particles 
    total_k=3*T_equal*(n-1)/2 
    vel_per_particle=np.zeros((n,3))
    for axis in range(3):
        vel_per_particle[:,axis]=np.random.randn(n,)-0.5
    Mom_x_total=np.sum(vel_per_particle[:,0])
    Mom_y_total=np.sum(vel_per_particle[:,1])
    Mom_z_total=np.sum(vel_per_particle[:,2])
    Mom_x_avg=Mom_x_total/n
    Mom_y_avg=Mom_y_total/n
    Mom_z_avg=Mom_z_total/n
    vel_per_particle=vel_per_particle-np.array([Mom_x_avg,Mom_y_avg,Mom_z_avg]).reshape(1,3)
    k_avg_init=0.5*(1/n)*np.sum(np.sum(vel_per_particle**2,axis=1),axis=0)
    k_avg_T_eq=total_k/n
    scaling_ratio=np.sqrt(k_avg_T_eq/k_avg_init)
    vel_per_particle=vel_per_particle*scaling_ratio
    return vel_per_particle

# @numba.njit
def Kin_Eng(velocity):
    ##This function compute the average kinetic energy of the system at given the velocity
    #Input: velocity -- the velocities of all the particles
    #Output: Kinetic_sum -- the average kinetic energy of the system
    num=velocity.shape[0]
    Kinetic_per=0.5*np.sum(velocity**2,axis=1)
    Kinetic_avg=np.sum(Kinetic_per,axis=0)
    return Kinetic_avg

# @numba.njit
def LJ_potent_nondimen(position,r_cut,L):
    ##This function compute the nondimensional potential energy of the system at the given position
    #Input: position -- the position of all the particles at this instance
    #       r_cut -- the cutoff for the short-range force field
    #       L --  the size of the simulation cell
    #Output: np.sum(update_LJ) -- the potential energy of the system at this instance
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

# @numba.njit
def insta_pressure(L,T,position,r_cut,e_scale):
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
    pres_virial=np.zeros((num-1,1))
    for atom in range(num-1):
        position_relevent=position[atom:,:]
        position_other=position_relevent[1:,:]
        position_atom=position_relevent[0,:]
        #pbc rule 2
        separation=position_atom-position_other
        separation_new=pbc2(separation=separation,L=L)
        r_relat=np.sqrt(np.sum(separation_new**2,axis=1)).reshape(separation_new.shape[0],)
        force=[]
        active_r_relat=[]
        #get out the particles inside the r_limit
        for r0 in r_relat:
            if r0 <= r_cut:
               #active_r_relat.append(r0)
               force_num=-(24*r0**(-7)-48*r0**(-13))+dU_drcut
               force.append(force_num)
               active_r_relat.append(r0)    
        #get out the particles inside the r_cut
        active_amount=np.array(active_r_relat).shape[0]
        rijFij=np.array(active_r_relat).reshape(1,active_amount)@np.array(force).reshape(active_amount,1)
        pres_virial[atom,:]=rijFij
    pres_insta=pres_ideal+np.sum(pres_virial,axis=0)/(3*V)
    return pres_insta

#need to initialize the type before calling the function
# float_array = types.float64[:,:]
# @numba.njit()
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
      if atomID>15:
          print(atomID,atom, atom[:,0][0]==L)
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

#@numba.njit()
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

#@numba.njit
def concatDict(infodict):
  wholeDict=None
  for i,spd in infodict.items():
      if(wholeDict is None):
        wholeDict = copy.deepcopy(spd)
      else:
        wholeDict.concat(spd)
  return wholeDict

#@numba.njit()
def LJ_accel(position,neighb_x_0,r_cut,L, rank):
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

def data_saver(info, PE, KE, T_insta, P_insta, L, num_atoms,part_type,name,period,stop_step, r_c, ncore = 8, make_directory=True):
    iterations=str(stop_step)
    if make_directory == True:
        os.mkdir('results')
    # path_to_file_xyz="results/"+name+"position_last"+iterations+"stps"+".xyz"
    path_to_file_xyz = "results/"+name+"iter"+iterations+'core'+str(ncore)+'.txt'
    path_to_file_other = "results/"+name+"_Energy_Temp_Pres"+str(ncore)+'core'+".csv"
    path_to_file_summary = "results/"+name+"iter"+iterations+'core'+str(ncore)+"_summary"+".txt"
    p= open(path_to_file_xyz,"w")
    s= open(path_to_file_summary,'w')

    # #writing xyz file
    # comment = 'This is the position of the system during the last '+iterations+' steps'
    # # for atoms in info[-int(iterations):,:,:]:
    #     p.write("%s\n" % str(num_atoms))
    #     p.write("%s\n" % comment)
    #     for atom in atoms:
    #         # p.write(part_type)
    #         p.write("\t%s\n" % str(atom)[1:-2])

    #writing xyz file
    comment = 'This is the position of the system of the final step'
    # print(path_to_file_xyz)
    # for atoms in info[-int(iterations):,:,:]:
    # p.write("%s\n" % str(num_atoms))
    # p.write("%s\n" % comment)
    # for i in range(num_atoms):
    #     p.write("\t%s\n" % str(info[:]))
    # output = info[-1,:,:][np.argsort(info[-1,:,0])]
    output = info[-1,:,:]
    np.savetxt(path_to_file_xyz,output,fmt='%.6f')

    #writing other file
    other_dict={}
    other_dict['KE']=KE.reshape(KE.shape[0],)
    other_dict['PE']=PE.reshape(PE.shape[0],)
    other_dict['T_insta']=T_insta.reshape(T_insta.shape[0],)
    other_dict['P_insta']=P_insta.reshape(P_insta.shape[0],)
    other_df=pd.DataFrame.from_dict(other_dict)
    other_df.to_csv(path_to_file_other,index_label='step')

    #writing summary file
    s.write("Simulation Cell Size(unitless): "+str(L)+"\n")
    s.write("Simulation Particles Amount: "+str(num_atoms)+"\n")
    s.write("File Name: "+str(name)+"\n")
    s.write("Simulation Time: "+ str(period)+"\n")
    s.write("Simulation Step: "+str(stop_step)+"\n")
    s.write("Force Cut-off: "+str(r_c)+"\n")
    p.close()
    s.close()

