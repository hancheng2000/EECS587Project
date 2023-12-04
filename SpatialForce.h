#ifndef SF_H
#define SF_H

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <fstream>

std::vector<std::vector<double>> LJ_accel(const std::vector<std::vector<double>>& position,
                                         const std::vector<std::vector<double>>& neighb_x_0,
                                         double r_cut, double L) {
    int subcube_atoms = position.size();

    // Concatenate position and neighb_x_0
    std::vector<std::vector<double>> combined_position = position;
    combined_position.insert(combined_position.end(), neighb_x_0.begin(), neighb_x_0.end());

    int num = combined_position.size();
    std::vector<std::vector<double>> update_accel(subcube_atoms, std::vector<double>(3, 0.0));

    double dU_drcut = 48 * std::pow(r_cut, -13) - 24 * std::pow(r_cut, -7);

    for (int atom = 0; atom < subcube_atoms; ++atom) {
        std::vector<std::vector<double>> position_other;
        position_other.insert(position_other.end(), combined_position.begin(), combined_position.begin() + atom);
        position_other.insert(position_other.end(), combined_position.begin() + atom + 1, combined_position.end());

        std::vector<double> position_atom = combined_position[atom];
        std::vector<double> separation;
        for (size_t i = 0; i < position_other.size(); ++i) {
            for (size_t j = 0; j < position_atom.size(); ++j) {
                separation.push_back(position_atom[j] - position_other[i][j]);
            }
        }

        std::vector<double> separation_new = pbc2(separation, L);
        std::vector<double> r_relat;

        for (size_t i = 0; i < separation_new.size(); i += position_atom.size()) {
            double sum_squared = 0.0;
            for (size_t j = 0; j < position_atom.size(); ++j) {
                sum_squared += std::pow(separation_new[i + j], 2);
            }
            r_relat.push_back(std::sqrt(sum_squared));
        }

        std::vector<std::vector<double>> accel(r_relat.size(), std::vector<double>(3, 0.0));

        for (size_t i = 0; i < r_relat.size(); ++i) {
            if (r_relat[i] <= r_cut) {
                std::vector<double> separation_active_num = separation_new.begin() + i * position_atom.size();
                std::vector<double> vector_part;
                for (size_t j = 0; j < position_atom.size(); ++j) {
                    vector_part.push_back(separation_active_num[j] * (1 / r_relat[i]));
                }
                double scalar_part = 48 * std::pow(r_relat[i], -13) - 24 * std::pow(r_relat[i], -7) - dU_drcut;

                for (size_t j = 0; j < position_atom.size(); ++j) {
                    accel[i][j] = vector_part[j] * scalar_part;
                }
            }
        }

        for (size_t i = 0; i < accel.size(); ++i) {
            for (size_t j = 0; j < accel[i].size(); ++j) {
                update_accel[atom][j] += accel[i][j];
            }
        }
    }

    return update_accel;
}

#endif