from mpi4py import MPI
import numpy as np
import utils_force_decompose as ut
import vel_parallel as utp
import ForceDecomp as fd
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

    # MD simulation 
    for step in range(stop_step):
        if rank==0:
            # separate atoms to different processors  
            # Processor matrix p*p*1
            p = np.sqrt(MPI.COMM_WORLD)
            x = math.floor(rank / p)
            y = math.floor(rank % p)
            natoms = size_sim
            nx_per_processor = natoms / p
            ny_per_processor = natoms / p

            atoms_x = info[step,0:nx_per_processor,:]
            atoms_y = info[step,0:ny_per_processor,:]
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
                    comm.send(info[step,atoms_to_send_y_range[0]:math.floor((atoms_to_send_y_range[0]+atoms_to_send_y_range[1])/2),:],dest=i_rank,tag=1)
                elif x<y:
                    # sending the second half, e.g. 4567+23
                    comm.send(info[step,math.floor((atoms_to_send_y_range[0]+atoms_to_send_y_range[1])/2):atoms_to_send_y_range[1],:],dest=i_rank,tag=1)
                else:
                    # diagonal processors have full row and column atoms, e.g. 0123+0123
                    comm.send(info[step,atoms_to_send_y_range[0]:atoms_to_send_y_range[1],:],dest=i_rank,tag=1)
        else:
            atoms_x = comm.recv(source=0,tag=0)
            atoms_y = comm.recv(source=0,tag=1)
            atoms = np.concatenate((atoms_x, atoms_y),0)
        comm.barrier()


        # calculate forces
        force_per_processor, energy_per_processor = fd.LJ_energy_force(atoms_x, atoms_y, r_cut, L)
        force = force_per_processor
        energy = energy_per_processor
        # gather forces
        # Gather all the points into sqrt(MPI.COMM_WORLD) processors for position update
        p = int(np.sqrt(MPI.COMM_WORLD))
        x = math.floor(rank / p)
        y = math.floor(rank % p)
        # send the force to transposed matrix for gathering force
        if y>x:
            dest_rank = y * p + x
            comm.send(force,dest=dest_rank, tag=2)
            comm.send(energy,dest=dest_rank,tag=22)
        elif y<x:
            source_rank = y * p + x
            force_other = comm.recv(source = source_rank, tag=2)
            energy_other = comm.recv(source=source_rank,tag=22)
            force = np.vstack((force,force_other))
            energy = energy + energy_other
        elif y==x:
            assert force.shape[0] == force.shape[1]
            # multiplying by two here because when we sum up in the end, there is a factor of 2 in energy. 
            # The diagonal processors do not have transpose processor so we need to account for the 2 here
            energy = energy * 2 
        comm.barrier()

        # send the updated force matrix back to the transposed matrix and apply Newton third law
        if y<x:
            force_send = np.transpose(force) #Newton third law
            energy_send = energy
            dest_rank = y * p + x
            comm.send(force_send, dest = dest_rank, tag = 3)
            comm.send(energy_send, dest = dest_rank, tag = 33)
        elif y>x:
            source_rank = y * p + x
            force_new = comm.recv(source = source_rank, tag = 3)
            energy_new = comm.recv(source = source_rank, tag = 33)
            force = force_new
            energy = energy_new
        elif y==x:
            assert force.shape[0] == force.shape[1]
        comm.barrier()
        # gather the forces to the diagonal processor
        if x==y:
            subset_ranks = x * p + np.arange(0,p,1) 
            subset_comm_group = comm.Create_group(subset_ranks)
            subset_comm = comm.Create(subset_comm_group)
            force_all = np.zeros((force.shape[0],force.shape[1]))
            energy_all = 0
            assert force_all.shape[0] == force_all.shape[1]
            subset_comm.reduce(force,force_all,op=MPI.SUM, root=y)
            subset_comm.reduce(energy, energy_all, op=MPI.SUM, root=y)
            # sum up the force to be in the shape (natoms_per_processor, 1)
            final_force = np.sum(force_all,axis=1)
            final_energy = np.sum(energy_all)
            
            # update the velocity, acceleration and position
            atoms_x = utp.vel_Ver(atoms_x,final_force, dt,r_cut,L)
            # atoms_y = utp.vel_Ver(atoms_y,dt,r_cut,L)
        comm.barrier()
        # send the updated positions to the processor 0
        if rank==0:
            subset_ranks = np.arange(0,p,1) * (p + 1)
            subset_comm_group = comm.Create_group(subset_ranks)
            subset_comm = comm.Create(subset_comm_group)
            info_new = info[step+1,:,:].copy()
            info_new = subset_comm.gather(atoms_x,root=0)
            info[step+1,:,:] = info_new
            # calculate PE,KE,T,P
            PE=0
            subset_comm.reduce(energy_all,PE,op=MPI.SUM, root=0)
            PE = PE / 2
            KE[step+1,:] = ut.Kin_Eng(info[step+1,:,3:6])
            T_insta[step+1,:]=2*KE[step+1,:]*e_scale/(3*(size_sim-1)*k_B) #k
            P_insta[step+1,:]=ut.insta_pressure(L,T_insta[step+1],info[step+1,:,0:3],r_cut,e_scale) #unitless            
        comm.barrier()
    if rank==0:
      return info,PE,KE,T_insta,P_insta
            
