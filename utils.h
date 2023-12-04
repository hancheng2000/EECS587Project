#ifndef U_H
#define U_H

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <fstream>

using namespace std;

// class SpatialDomainData {
// public:
//     std::vector<std::vector<double>> P;
//     std::vector<std::vector<double>> V;
//     std::vector<std::vector<double>> A;

//     SpatialDomainData(std::vector<std::vector<double>> position = {},
//                       std::vector<std::vector<double>> velocity = {},
//                       std::vector<std::vector<double>> acceleration = {})
//         : P(position), V(velocity), A(acceleration) {}

//     friend std::ostream& operator<<(std::ostream& os, const SpatialDomainData& spd) {
//         os << "(P:";
//         for (const auto& p : spd.P) {
//             os << "[";
//             for (const auto& elem : p) {
//                 os << elem << ",";
//             }
//             os << "]";
//         }
//         os << ", V:";
//         for (const auto& v : spd.V) {
//             os << "[";
//             for (const auto& elem : v) {
//                 os << elem << ",";
//             }
//             os << "]";
//         }
//         os << ", A:";
//         for (const auto& a : spd.A) {
//             os << "[";
//             for (const auto& elem : a) {
//                 os << elem << ",";
//             }
//             os << "]";
//         }
//         os << ")";
//         return os;
//     }

//     void concat(const SpatialDomainData& other) {
//         P.insert(P.end(), other.P.begin(), other.P.end());
//         V.insert(V.end(), other.V.begin(), other.V.end());
//         A.insert(A.end(), other.A.begin(), other.A.end());
//     }
// };

std::vector<std::vector<double>> pbc1(const std::vector<std::vector<double>>& position, double L) {
    std::vector<std::vector<double>> position_new(position.size(), std::vector<double>(3, 0.0));

    for (size_t i = 0; i < position.size(); ++i) {
        const auto& position_ind = position[i];
        std::vector<double> position_empty(3, 0.0);

        for (size_t j = 0; j < 3; ++j) {
            double position_axis = position_ind[j];
            double position_axis_new;

            if (position_axis < 0) {
                position_axis_new = position_axis + L;
            } else if (position_axis > L) {
                position_axis_new = position_axis - L;
            } else {
                position_axis_new = position_axis;
            }

            position_empty[j] = position_axis_new;
        }

        position_new[i] = position_empty;
    }

    return position_new;
}

std::vector<std::vector<double>> pbc2(const std::vector<std::vector<double>>& separation, double L) {
    std::vector<std::vector<double>> separation_new(separation.size(), std::vector<double>(3, 0.0));

    for (size_t i = 0; i < separation.size(); ++i) {
        const auto& separation_ind = separation[i];
        std::vector<double> separation_empty(3, 0.0);

        for (size_t j = 0; j < 3; ++j) {
            double separation_axis = separation_ind[j];
            double separation_axis_new;

            if (separation_axis < -L / 2) {
                separation_axis_new = separation_axis + L;
            } else if (separation_axis > L / 2) {
                separation_axis_new = separation_axis - L;
            } else {
                separation_axis_new = separation_axis;
            }

            separation_empty[j] = separation_axis_new;
        }

        separation_new[i] = separation_empty;
    }

    return separation_new;
}

std::vector<std::vector<double>> random_vel_generator(int n, double T_equal, double e_scale) {
    double k_B = 1.38064852e-23;
    double total_k = 3 * T_equal * (n - 1) / 2;
    std::vector<std::vector<double>> vel_per_particle(n, std::vector<double>(3, 0.0));

    for (size_t axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < n; ++i) {
            vel_per_particle[i][axis] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    double Mom_x_total = 0.0;
    double Mom_y_total = 0.0;
    double Mom_z_total = 0.0;

    for (int i = 0; i < n; ++i) {
        Mom_x_total += vel_per_particle[i][0];
        Mom_y_total += vel_per_particle[i][1];
        Mom_z_total += vel_per_particle[i][2];
    }

    double Mom_x_avg = Mom_x_total / double(n);
    double Mom_y_avg = Mom_y_total / double(n);
    double Mom_z_avg = Mom_z_total / double(n);

    double sum_vel=0.0;

    for (int i = 0; i < n; ++i) {
        vel_per_particle[i][0] -= Mom_x_avg;
        vel_per_particle[i][1] -= Mom_y_avg;
        vel_per_particle[i][2] -= Mom_z_avg;
        sum_vel += vel_per_particle[i][0]**2 + vel_per_particle[i][1]**2 + vel_per_particle[i][2]**2;
    }


    // double k_avg_init = 0.5 * (1.0 / n) * std::accumulate(vel_per_particle.begin(), vel_per_particle.end(), 0.0,
    //                                                       [](double acc, const std::vector<double>& v) {
    //                                                           return acc + std::accumulate(v.begin(), v.end(), 0.0,
    //                                                                                         [](double sub_acc, double elem) {
    //                                                                                             return sub_acc + elem * elem;
    //                                                                                         });
    //                                                       });
    double k_avg_init = 0.5 * (1.0 / double(n)) * sum_vel;
    double k_avg_T_eq = total_k / n;
    double scaling_ratio = std::sqrt(k_avg_T_eq / k_avg_init);

    for (int i = 0; i < n; ++i) {
        vel_per_particle[i][0] *= scaling_ratio;
        vel_per_particle[i][1] *= scaling_ratio;
        vel_per_particle[i][2] *= scaling_ratio;
    }

    return vel_per_particle;
}

double Kin_Eng(const std::vector<std::vector<double>>& velocity) {
    double Kinetic_avg = 0.0;

    for (const auto& v : velocity) {
        double Kinetic_per = 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        Kinetic_avg += Kinetic_per;
    }

    return Kinetic_avg;
}

double LJ_potent_nondimen(const std::vector<std::vector<double>>& position, double r_cut, double L) {
    int num = position.size();
    // std::vector<std::vector<double>> update_LJ(num - 1, std::vector<double>(1, 0.0));
    std::vector<double> update_LJ(num-1);

    // Fixed value for a certain r_limit
    double dU_drcut = 24 * pow(r_cut, -7) - 48 * pow(r_cut, -13);
    double U_rcut = 4 * (pow(r_cut, -12) - pow(r_cut, -6));

    for (int atom = 0; atom < num - 1; ++atom) {
        // std::vector<std::vector<double>> position_relevent(position.begin() + atom, position.end());
        // std::vector<std::vector<double>> position_other(position_relevent.begin() + 1, position_relevent.end());
        std::vector<std::vector<double>> position_other(position.begin() + atom + 1, position.end());

        // PBC rule2
        std::vector<std::vector<double>> separation(position.begin() + atom + 1, position.end());
        for (size_t i = 0; i < position_other.size(); ++i) {
            for (size_t j = 0; j < position[0].size(); ++j) {
                seperation[i][j] = position[atom][j] - position_other[i][j];
                // separation.push_back(position_relevent[0][j] - position_other[i][j]);
            }
        }

        std::vector<double> separation_new = pbc2(separation, L);
        std::vector<double> r_relat;
        // compute rij
        for (size_t i = 0; i < separation_new.size(); ++i) {
            double sum_squared = 0.0;
            for (size_t j = 0; j < position_relevent[0].size(); ++j) {
                sum_squared += pow(separation_new[i][j], 2);
            }
            r_relat.push_back(std::sqrt(sum_squared));
        }

        // std::vector<double> LJ;
        double LJ = 0.0;
        // Get out the particles inside the r_limit
        for (double r0 : r_relat) {
            if (r0 <= r_cut) {
                // double LJ_num = 4 * std::pow(r0, -12) - 4 * std::pow(r0, -6) - U_rcut - (r0 - r_cut) * dU_drcut;
                LJ += 4 * std::pow(r0, -12) - 4 * std::pow(r0, -6) - U_rcut - (r0 - r_cut) * dU_drcut;
                // LJ.push_back(LJ_num);
            }
            // update_LJ[atom] = std::accumulate(LJ.begin(), LJ.end(), 0.0);
            update_LJ[atom] = LJ;
        }
    }

    return std::accumulate(update_LJ.begin(), update_LJ.end(), 0.0);
}



#endif