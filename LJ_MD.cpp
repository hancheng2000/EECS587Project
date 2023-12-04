#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "utils.h"

const double k_B = 1.38064852e-23;

// Assuming you have equivalent functions in your utils and utils_parallel namespaces
namespace utils {
    // Function declarations for LJ_potent_nondimen, Kin_Eng, insta_pressure, random_vel_generator, and other necessary functions
}

namespace utils_parallel {
    // Function declarations for par_worker_vel_ver_pre and vel_Ver
}

void LJ_MD(int subdiv[3], std::vector<std::vector<double>>& position_init, double dt, int stop_step,
           std::vector<std::vector<double>>& accel_init, double r_cut, double L, double T_eq,
           double e_scale, double sig) {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialization
    if (rank == 0) {
        int size_sim = position_init.size();
        // velocity initialization
        std::vector<std::vector<double>> x_dot_init = utils::random_vel_generator(size_sim, T_eq, e_scale);
        // Initialize PE, KE, T_insta, P_insta
        std::vector<double> PE(stop_step + 1, 0.0);
        std::vector<double> KE(stop_step + 1, 0.0);
        std::vector<double> T_insta(stop_step + 1, 0.0);
        std::vector<double> P_insta(stop_step + 1, 0.0);
        // Zero step values
        PE[0] = utils::LJ_potent_nondimen(position_init, r_cut, L);
        KE[0] = utils::Kin_Eng(x_dot_init);
        T_insta[0] = 2 * KE[0] * e_scale / (3 * (size_sim - 1) * k_B);
        P_insta[0] = utils::insta_pressure(L, T_insta[0], info[0], r_cut, e_scale);
        
        // // Initialize info matrix
        // std::vector<std::vector<std::vector<double>>> info(stop_step + 1, std::vector<std::vector<double>>(size_sim, std::vector<double>(9)));
        // for (int i = 0; i < size_sim; ++i) {
        //     info[0][i] = position_init[i];
        //     info[0][i].insert(info[0][i].end(), x_dot_init[i].begin(), x_dot_init[i].end());
        //     info[0][i].insert(info[0][i].end(), accel_init[i].begin(), accel_init[i].end());
        // }

        // // Convert info[0] to infotodic
        // auto infotodic = utils::cell_to_obj(info[0], subdiv[0], subdiv[1], subdiv[2], L);




    } else {
        // For non-rank 0 processes
        // You might need to initialize variables here
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Broadcast infotodic from rank 0 to all processes
    // Note: Assuming that infotodic is a simple vector or array
    MPI_Bcast(&infotodic[0], infotodic.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int step = 0; step < stop_step; ++step) {
        if (rank != 0) {
            // Call the vel_verlet parallel function
            utils_parallel::par_worker_vel_ver_pre(infotodic, dt, r_cut, L);
        } else {
            // Call the vel_Ver function on rank 0
            auto info_temp = utils_parallel::vel_Ver(infotodic, dt, r_cut, L);
            std::vector<std::vector<double>> tmp = utils::concatDict(info_temp);
            info[step + 1] = tmp;
            // Update cubes and make sure atoms are in the right cubes
            infotodic = utils::cell_to_obj(info[step + 1], subdiv[0], subdiv[1], subdiv[2], L);
            // Calculate and store PE, KE, T_insta, P_insta
            PE[step + 1] = utils::LJ_potent_nondimen(info[step + 1], r_cut, L);
            KE[step + 1] = utils::Kin_Eng(info[step + 1]);
            T_insta[step + 1] = 2 * KE[step + 1] * e_scale / (3 * (size_sim - 1) * k_B);
            P_insta[step + 1] = utils::insta_pressure(L, T_insta[step + 1], info[step + 1], r_cut, e_scale);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // Broadcast infotodic from rank 0 to all processes
        MPI_Bcast(&infotodic[0], infotodic.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    if (rank == 0) {
        // Return values for rank 0
        // You may need to adjust the return type or structure
        return {info, PE, KE, T_insta, P_insta};
    }
}

int main() {
    // Example usage
    int subdiv[3] = {10, 10, 10};
    std::vector<std::vector<double>> position_init = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
    double dt = 0.001;
    int stop_step = 100;
    std::vector<std::vector<double>> accel_init = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    double r_cut = 2.5;
    double L = 10.0;
    double T_eq = 300.0;
    double e_scale = 1.0;
    double sig = 1.0;

    auto result = LJ_MD(subdiv, position_init, dt, stop_step, accel_init, r_cut, L, T_eq, e_scale, sig);

    // Process the result as needed
    return 0;
}