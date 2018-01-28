/*
 * Copyright (c) 2017 Ivan Iakoupov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef HAMILTONIANPARAMS_H
#define HAMILTONIANPARAMS_H

#include "Eigen/Dense"
#include <vector>

// For the 2 mm long ensemble with linewidth
// of Gamma = 2 * pi * 10 s^(-1) we get
// (3e8 m/s) / (2 mm) / (2*pi*10e6 (1/s)) = 2387
// The exact value does not matter that much.
// For a long time this was set to 50 with no
// visible difference in the results of the
// simulations.
#define DEFAULT_C_OVER_L_VALUE 2000

enum class EvolutionMethod {
    Default,
    RK4,
    MatrixExp
};

struct HamiltonianParams
{
    bool adiabaticallyGoToRegularEit;
    bool adiabaticallyGoToCounterpropagatingEit;
    bool counterpropagating;
    bool putPhasesInTheClassicalDrive;
    bool fourth_level;
    std::complex<double> input_electric_field_R_factor;
    std::complex<double> input_electric_field_L_factor;
    bool use_input_electric_field_data;
    bool disable_classical_drive;
    bool putImpurity;
    bool randomAtomPositions;
    bool classicalDriveOppositeDetunings;
    int impurity_position;
    std::vector<double> atom_positions;
    std::vector<std::complex<double>> E_R_input_data;
    std::vector<std::complex<double>> E_L_input_data;
    std::vector<double> intermediate_times;
    double rydberg_interaction;
    int NAtoms;
    int intermediate_spinwave_snapshots;
    std::vector<double> deltaValues;
    double delta;
    double Deltac;
    double Delta2;
    double g1d;
    double Omega;
    double gridSpacing;
    double kd_ensemble;
    double cOverLGamma;
    double inputE_width;
    double inputE_mean;
    EvolutionMethod evolutionMethod;
    HamiltonianParams() : adiabaticallyGoToRegularEit(false),
                          adiabaticallyGoToCounterpropagatingEit(false),
                          counterpropagating(false),
                          putPhasesInTheClassicalDrive(false),
                          fourth_level(false),
                          input_electric_field_R_factor(0),
                          input_electric_field_L_factor(0),
                          use_input_electric_field_data(false),
                          disable_classical_drive(false),
                          putImpurity(false),
                          randomAtomPositions(false),
                          classicalDriveOppositeDetunings(false),
                          impurity_position(-1),
                          rydberg_interaction(0),
                          intermediate_spinwave_snapshots(1000),
                          cOverLGamma(DEFAULT_C_OVER_L_VALUE),
                          evolutionMethod(EvolutionMethod::MatrixExp)
    {}
};

inline double position_from_index(int i, double step)
{
    return i*step;
}

inline void
fill_in_regular_atomic_positions(double *atom_positions,
                                 int NAtoms, double gridSpacing)
{
    for (int i = 0; i < NAtoms; ++i) {
        const double distance = position_from_index(i, gridSpacing);
        atom_positions[i] = distance;
    }
}

inline double grid_spacing(int NAtoms, double L)
{
    return L/NAtoms;
}

#endif // HAMILTONIANPARAMS_H
