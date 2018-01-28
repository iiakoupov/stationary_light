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

#ifndef ENSEMBLE_SCATTERING_H
#define ENSEMBLE_SCATTERING_H

#include "ensemble_scattering_common.h"

#include <complex>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/StdVector"

#define ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS     (1 << 0)
#define ENSEMBLE_SCATTERING_DUAL_V_ATOMS              (1 << 1)
#define ENSEMBLE_SCATTERING_RYDBERG                   (1 << 2)

inline int impurity_position_close_to_middle(int NAtoms, int periodLength)
{
    int half = NAtoms/2;
    int modulo = half % periodLength;
    return half - modulo;
}

inline int find_period_length_from_kd(double kd_ensemble)
{
    const double kd_ensemble_reduced = std::fmod(kd_ensemble, 1);
    const double num_periods_d = 1.0/kd_ensemble_reduced;
    const int num_periods = static_cast<int>(num_periods_d);
    if (static_cast<double>(num_periods) == num_periods_d) {
        return num_periods;
    }
    return -1;
}

class EnsembleScattering
{
    int m_NAtoms;
    double m_kdEnsemble;
    double m_kdOmega;
    double m_delta;
    double m_Deltac;
    double m_g1d;
    double m_Omega;
    double m_C_6;
    int m_flags;
    unsigned long int m_randomSeed;
    unsigned long int m_numRealizationsToSkip;
    std::vector<double> m_atomPositions;
    std::vector<double> m_OmegaStrengths;
    std::vector<std::complex<double>> m_phaseFactors;
    std::vector<std::complex<double>> m_OmegaPhaseFactors;
    std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>> m_transferMatrices2;
    std::vector<Eigen::Matrix4cd, Eigen::aligned_allocator<Eigen::Matrix4cd>> m_transferMatrices4;

public:
    EnsembleScattering() = delete;
    EnsembleScattering(const EnsembleScattering &other);
    EnsembleScattering& operator=(const EnsembleScattering&) = delete;
    EnsembleScattering(int flags);

    const std::vector<double> &atomPositions() { return m_atomPositions; }
    double kdEnsemble() const { return m_kdEnsemble; }
    const std::vector<std::complex<double>> &phaseFactors() { return m_phaseFactors; }
    void initializeRandomGenerator();
    int flags() const { return m_flags; }
    bool randomAtomPositions() const {
        return m_flags & ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS;
    }
    unsigned long int randomSeed() const { return m_randomSeed; }
    const std::vector<double> &OmegaStrengths() { return m_OmegaStrengths; }
    void fillAtomArraysNewValues(int NAtoms,
                                 double kd_ensemble,
                                 double kd_Omega);
    void fillAtomArraysNewValues(int NAtoms,
                                 double kd_ensemble,
                                 double kd_Omega,
                                 double delta,
                                 double Deltac,
                                 double g1d,
                                 double Omega);
    void fillAtomArrays();
    void fillTransferMatrixArrayNewValues(double delta,
                                          double Deltac,
                                          double g1d,
                                          double Omega);
    void fillTransferMatrixArray();
    void fillTransferMatrixArrayDualV();
    void setC_6(double C_6);
    void setRandomSeed(unsigned long seed, unsigned long numRealizationsToSkip = 0);

    RandTCoefficients
    rAndTCoefficientsLambda(double delta, double kd_ensemble,
                            double kd_Omega, double Deltac,
                            double g1d, double Omega, int NAtoms,
                            int impurityPosition);
    RandTCoefficients
    rAndTCoefficientsLambdaRydberg(double delta, double kd_ensemble,
                            double kd_Omega, double Deltac,
                            double g1d, double Omega, int NAtoms,
                            int impurityPosition);

    // This function is only thread-safe in the case
    // when the only thing that is changing is impurityPosition,
    // and fillTransferMatrixArrayNewValues() was called prior
    // to calling this one with different impurityPosition arguments.
    RandTCoefficients
    rAndTCoefficientsDualV(double delta, double kd_ensemble,
                           double kd_Omega, double Deltac,
                           double g1d, double Omega, int NAtoms,
                           int impurityPosition);

    // The above comment doesn't apply to this function, since
    // when using Rydberg interaction we cannot use the tricks
    // with caching most of the transfer matrices and have to
    // compute them anew for each impurity position.
    RandTCoefficients
    rAndTCoefficientsDualVRydberg(double delta, double kd_ensemble,
                           double kd_Omega, double Deltac,
                           double g1d, double Omega, int NAtoms,
                           int impurityPosition);
    RandTCoefficients
    operator()(double delta, double kd_ensemble, double kd_Omega,
               double Deltac, double g1d, double Omega,
               int NAtoms, int impurityPosition);
};

#endif // ENSEMBLE_SCATTERING_H
