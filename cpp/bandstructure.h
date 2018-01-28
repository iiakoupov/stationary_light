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

#ifndef BANDSTRUCTURE_H
#define BANDSTRUCTURE_H

#include "hamiltonian_params.h"

#include <complex>
#include <vector>

template <typename Scalar>
inline std::complex<Scalar>
acos_on_branch(std::complex<Scalar> y, int branch)
{
    // branch == 0 is the principal one with values in [0, pi).
    // branch == 1 has values in [pi, 2*pi) and is defined as
    //             2*pi-arccos(y)
    // branch == 2 has values in [2*pi, 3*pi) and is defined as
    //             arccos(y)+2*pi
    // branch == 3 has values in [3*pi, 4*pi) and is defined as
    //             (2*pi-arccos(y))+2*pi
    // etc.

    //std::cout << "branch = " << branch << std::endl;
    if (branch % 2 == 0) {
        return std::acos(y) + Scalar(branch*M_PI);
    } else {
        return (Scalar(2*M_PI) - std::acos(y)) + Scalar((branch/2)*2*M_PI);
    }
}

std::complex<double> qd_eit_standing_wave(std::complex<double> delta, double Deltac,
                                          double Omega, double g1d,
                                          double gprime, unsigned periodLength,
                                          double phaseShift);

#define KD_BLOCH_BAND_RANDOM_LAMBDA_ATOMS       0
#define KD_BLOCH_BAND_RANDOM_DUAL_V_ATOMS       1
#define KD_BLOCH_BAND_RANDOM_DUAL_COLOR_ATOMS   2

enum class LevelScheme {
    LambdaType,
    dualV,
    dualColor
};

class QDOnBlochBandRandom
{
    bool m_aboutToSwitchToAHigherBranch;
    bool m_aboutToSwitchToALowerBranch;
    bool m_regularPlacement;
    double m_deltaLast;
    double m_kdLast;
    int m_NAtoms;
    int m_classicalDrivePeriods;
    LevelScheme m_level_scheme;
    std::vector<double> m_phases;
    int m_randomSeed;
    int m_branchShift;
    Eigen::MatrixXcd m_last_eigenvector_matrix;
public:
    QDOnBlochBandRandom(int classicalDrivePeriods, LevelScheme level_scheme,
                        bool regularPlacement);

    std::vector<std::complex<double>> qd(
            int NAtoms, int NDeltad,
            std::complex<double> Delta, double Deltac, double Deltad,
            double Omega, double g1d, double shift, double distributionWidth)
    {
        switch (m_level_scheme) {
        case LevelScheme::dualV:
            return qdDualV(NAtoms, NDeltad, Delta, Deltac, Deltad, Omega, g1d, shift, distributionWidth);
        case LevelScheme::dualColor:
            return qdDualColor(NAtoms, NDeltad, Delta, Deltac, Deltad, Omega, g1d, shift, distributionWidth);
        case LevelScheme::LambdaType:
            return qdLambda(NAtoms, NDeltad, Delta, Deltac, Deltad, Omega, g1d, shift, distributionWidth);
        default:
            assert(0 && "Unknown level scheme");
        }
    }

    std::vector<std::complex<double>>
    qdLambda(int NAtoms, int NDeltad,
             std::complex<double> delta, double Deltac, double Deltad,
             double Omega, double g1d, double shift,
             double distributionWidth);

    std::vector<std::complex<double>>
    qdDualV(int NAtoms, int NDeltad,
            std::complex<double> delta, double Deltac, double Deltad,
            double Omega, double g1d, double shift,
            double distributionWidth);

    std::vector<std::complex<double>>
    qdDualColor(int NAtoms, int NDeltad,
                std::complex<double> delta, double Deltac, double Deltad,
                double Omega, double g1d, double shift,
                double distributionWidth);
    double meanKD() const;
    int randomSeed() { return m_randomSeed; }
    void setRandomSeed(int randomSeed, int NAtoms);
};

#endif // BANDSTRUCTURE_H
