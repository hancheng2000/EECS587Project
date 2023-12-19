from mpi4py import MPI
import numpy as np
import utils as ut
import utils_parallel as utp
def LJ_MD(subdiv,position_init,dt,stop_step,accel_init,r_cut,L,T_eq,e_scale,sig):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #initialization
    if rank == 0:
        k_B=1.38064852*10**(-23)
        size_sim=position_init.shape[0]
        x_dot_init=ut.random_vel_generator(size_sim,T_eq,e_scale)
        #initialize PE, KE, T_insta, P_insta, Momentum
        PE=np.zeros((stop_step+1,1))
        KE=np.zeros((stop_step+1,1))
        T_insta=np.zeros((stop_step+1,1))
        P_insta=np.zeros((stop_step+1,1))      
        #zero step value
        PE[0,:]=ut.LJ_potent_nondimen(info[0,:,0:3],r_cut=r_cut,L=L)
        KE[0,:]=ut.Kin_Eng(info[0,:,3:6])
        T_insta[0,:]=2*KE[0,:]*e_scale/(3*(size_sim-1)*k_B) #k
        P_insta[0,:]=ut.insta_pressure(L,T_insta[0],info[0,:,0:3],r_cut,e_scale) #unitless      
        #initialize the info matrix
        info=np.zeros((stop_step+1,size_sim,9))
        info[0,:,:]=np.concatenate((position_init,x_dot_init,accel_init),axis=1)
        # infotodic=ut.cell_to_obj((info[0,:,:]),subdiv[0],subdiv[1],subdiv[2],L)

        # separate atoms to different processors  
        # Processor matrix p*p*1
        p = np.sqrt(comm.world)
        x = math.floor(rank / p)
        y = math.floor(rank % p)
        natoms = size_sim
        nx_per_processor = natoms / p
        ny_per_processor = natoms / p

        atoms_x = info[0,0:nx_per_processor,:]
        atoms_y = info[0,0:ny_per_processor,:]
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
            if x>y:
                # sending the first half, e.g. 4567+01 (supposing each core has 4 x row atoms and 4 column atoms)
                comm.send(info[atoms_to_send_y_range[0]:math.floor((atoms_to_send_y_range[0]+atoms_to_send_y_range[1])/2),:],dest=i_rank,tag=1)
            elif x<y:
                # sending the second half, e.g. 4567+23
                comm.send(info[math.floor((atoms_to_send_y_range[0]+atoms_to_send_y_range[1])/2):atoms_to_send_y_range[1],:],dest=i_rank,tag=1)
            else:
                # diagonal processors have full row and column atoms, e.g. 0123+0123
                comm.send(info[atoms_to_send_y_range[0]:atoms_to_send_y_range[1],:],dest=i_rank,tag=1)
    else:
        atoms_x = comm.recv(source=0,tag=0)
        atoms_y = comm.recv(source=0,tag=1)
        atoms = np.concatenate((atoms_x, atoms_y),0)
    comm.barrier()

    # TODO:MD simulation (yet to be changed)
    for step in range(stop_step):
        if rank!=0:
          #call the vel_verlet parallel function
          utp.par_worker_vel_ver_pre(infotodic,dt,r_cut,L)
        else:
          info_temp=utp.vel_Ver(infodict=infotodic,dt=dt,r_limit=r_cut,L=L)
          tmp=ut.concatDict(info_temp)
          info[step+1,:,:]=np.concatenate((tmp.P,tmp.V,tmp.A),1)
          #UPDATE CUBES MAKE SURE ATOMS ARE IN RIGHT CUBES
          infotodic=ut.cell_to_obj(info[step+1,:,:],subdiv[0],subdiv[1],subdiv[2],L)
          #calculate and store PE, KE, T_insta, P_insta
          PE[step+1,:]=ut.LJ_potent_nondimen(info[step+1,:,0:3],r_cut=r_cut,L=L)
          KE[step+1,:]=ut.Kin_Eng(info[step+1,:,3:6])
          T_insta[step+1,:]=2*KE[step+1,:]*e_scale/(3*(size_sim-1)*k_B) #k
          P_insta[step+1,:]=ut.insta_pressure(L,T_insta[step+1],info[step+1,:,0:3],r_cut,e_scale) #unitless
        comm.barrier()
        infotodic=comm.bcast(infotodic,root=0)
    if rank==0:
      return info,PE,KE,T_insta,P_insta
