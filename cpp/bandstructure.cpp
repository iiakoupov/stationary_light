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

#include "bandstructure.h"

#include "urandom.h"

#include "dual_v_scattering.h"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

#include <iostream>

namespace {
inline std::complex<double> beta3(std::complex<double> delta, double Deltac,
                                  double Omega, double g1d, double gprime)
{
    const std::complex<double> I(0, 1);
    const std::complex<double> Delta = delta+Deltac;
    return (g1d*delta)/((-2.0*I*Delta+gprime)*delta+2.0*I*std::pow(Omega,2));
}

inline std::complex<double> beta2(std::complex<double> Delta, double g1d,
                                  double gprime)
{
    const std::complex<double> I(0, 1);
    return g1d/(gprime - 2.0*I*Delta);
}

inline Eigen::Matrix2cd scattering_matrix(std::complex<double> beta)
{
    Eigen::Matrix2cd ret = Eigen::Matrix2cd::Zero();
    ret(0,0) = 1.0-beta;
    ret(0,1) = -beta;
    ret(1,0) = beta;
    ret(1,1) = 1.0+beta;
    return ret;
}

inline Eigen::Matrix2cd free_matrix(double kd)
{
    const std::complex<double> I(0, 1);
    const std::complex<double> kd_exp = std::exp(I*kd);
    Eigen::Matrix2cd ret = Eigen::Matrix2cd::Zero();
    ret(0,0) = kd_exp;
    ret(0,1) = 0;
    ret(1,0) = 0;
    ret(1,1) = std::conj(kd_exp);
    return ret;
}

inline Eigen::Matrix4cd free_matrix4(double kd)
{
    const std::complex<double> I(0, 1);
    const std::complex<double> kd_exp = std::exp(I*kd);
    Eigen::Matrix4cd ret = Eigen::Matrix4cd::Zero();
    ret(0,0) = kd_exp;
    ret(1,1) = kd_exp;
    ret(2,2) = std::conj(kd_exp);
    ret(3,3) = std::conj(kd_exp);
    return ret;
}

inline Eigen::MatrixXcd free_matrix_N(double kd, int Nmodes)
{
    // We assume here that all the modes have the same frequency
    // which is not true for dual-color strictly speaking.
    // We hope that the higher order modes do not contribute
    // much and neglecting the free space dispersion for them
    // is fine.
    const std::complex<double> I(0, 1);
    const std::complex<double> kd_exp = std::exp(I*kd);
    Eigen::MatrixXcd ret = Eigen::MatrixXcd::Zero(2*Nmodes, 2*Nmodes);
    for (int i = 0; i < Nmodes; ++i) {
        ret(i,i) = kd_exp;
        ret(i+Nmodes,i+Nmodes) = std::conj(kd_exp);
    }
    return ret;
}

inline Eigen::Matrix2cd M_trans_stationary_light_lambda(
        const std::vector<double> &phases, int NAtoms,
        double mean_kd,
        std::complex<double> delta, double Deltac, double Omega,
        double g1d, double g)
{
    const std::complex<double> I(0, 1);
    Eigen::Matrix2cd ret = Eigen::Matrix2cd::Identity();
    double last_phase = 0;
    for (int i = 0; i < NAtoms; ++i) {
        const double total_phase = phases[i];
        ret = free_matrix(total_phase-last_phase)*ret;
        const double OmegaAtom = Omega*std::cos(total_phase);
        ret = scattering_matrix(beta3(delta, Deltac, OmegaAtom, g1d, g))*ret;
        last_phase = total_phase;
    }
    const double final_total_phase = NAtoms*mean_kd*M_PI;
    const double phase_diff = final_total_phase-last_phase;
    ret = free_matrix(phase_diff)*ret;
    return ret;
}

inline Eigen::Matrix4cd M_trans_stationary_light_dualv(
        const std::vector<double> &phases, int NAtoms,
        double mean_kd,
        std::complex<double> delta, double Deltac, double Omega,
        double g1d)
{
    Eigen::Matrix4cd ret = Eigen::Matrix4cd::Identity();
    const std::complex<double> I(0, 1);
    double last_phase = 0;
    for (int i = 0; i < NAtoms; ++i) {
        const double total_phase = phases[i];
        ret = free_matrix4(total_phase-last_phase)*ret;
        const std::complex<double> OmegaPhaseFactor = std::exp(2.0*I*total_phase);
        std::complex<double> rpp;
        std::complex<double> tpp;
        std::complex<double> rpm;
        std::complex<double> tpm;
        std::complex<double> rmm;
        std::complex<double> tmm;
        std::complex<double> rmp;
        std::complex<double> tmp;
        dualv_scattering_coefficients(rpp, tpp, rmp, tmp, rpm, tpm, rmm, tmm,
                                      Deltac, Deltac, delta, g1d,
                                      Omega, Omega, OmegaPhaseFactor);
        ret = scattering_matrix4_beta(rpp, rmp, rpm, rmm)*ret;
        last_phase = total_phase;
    }
    const double final_total_phase = NAtoms*mean_kd*M_PI;
    const double phase_diff = final_total_phase-last_phase;
    ret = free_matrix4(phase_diff)*ret;
    return ret;
}

inline Eigen::MatrixXcd scattering_matrixN_beta(Eigen::MatrixXcd S_r)
{
    const int Nmodes = S_r.rows();
    assert(Nmodes == S_r.cols() && "S_r is not a square matrix!");

    Eigen::MatrixXcd S_t = Eigen::MatrixXcd::Identity(Nmodes,Nmodes) + S_r;

    Eigen::FullPivLU<Eigen::MatrixXcd> lu(S_t);
    Eigen::MatrixXcd B = -lu.solve(S_r);

    Eigen::MatrixXcd ret = Eigen::MatrixXcd::Zero(2*Nmodes,2*Nmodes);

    ret.block(0,0,Nmodes,Nmodes) = Eigen::MatrixXcd::Identity(Nmodes,Nmodes) - B;
    ret.block(0,Nmodes,Nmodes,Nmodes) = -B;
    ret.block(Nmodes,0,Nmodes,Nmodes) = B;
    ret.block(Nmodes,Nmodes,Nmodes,Nmodes) = Eigen::MatrixXcd::Identity(Nmodes,Nmodes) + B;
    return ret;
}

inline int Nmodes_from_NDeltad(int NDeltad)
{
    return 2*NDeltad + 1;
}

inline Eigen::MatrixXcd dual_color_S_r(
        int NDeltad,
        double Deltac, double Deltad, std::complex<double> delta, double g1d, double Omega,
        std::complex<double> OmegaPhaseFactor)
{
    const std::complex<double> I(0,1);
    const std::complex<double> Delta = Deltac + delta;
    const double Omega2 = Omega*Omega;
    const double gprime = 1-g1d;
    const int Nmodes = Nmodes_from_NDeltad(NDeltad);
    const std::complex<double> Omega_cur = 0.5*Omega*(OmegaPhaseFactor+std::conj(OmegaPhaseFactor));
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(Nmodes, Nmodes);
    for (int n = -NDeltad; n <= NDeltad; ++n) {
        const int nPlusTwo = n+2;
        const int nMinusTwo = n-2;
        const int index = Nmodes-(n+NDeltad)-1;
        M(index, index) = -n*Deltad+Delta+0.5*I
                            +(0.25*Omega2)/((n+1)*Deltad-delta)
                            +(0.25*Omega2)/((n-1)*Deltad-delta);
        const int indexPlusTwo = Nmodes-(nPlusTwo+NDeltad)-1;
        if (indexPlusTwo >= 0 && indexPlusTwo < Nmodes) {
            M(index,indexPlusTwo) = (0.25*Omega2*OmegaPhaseFactor)/((n+1)*Deltad-delta);
        }
        const int indexMinusTwo = Nmodes-(nMinusTwo+NDeltad)-1;
        if (indexMinusTwo >= 0 && indexMinusTwo < Nmodes) {
            M(index,indexMinusTwo) = (0.25*Omega2*std::conj(OmegaPhaseFactor))/((n-1)*Deltad-delta);
        }
    }
    Eigen::MatrixXcd S_r = -I*0.5*g1d*M.inverse();
    return S_r;
}

inline Eigen::MatrixXcd M_trans_stationary_light_dual_color(
        const std::vector<double> &phases, int NAtoms,
        int NDeltad, double mean_kd, std::complex<double> delta,
        double Deltac, double Deltad, double Omega, double g1d)
{
    const int Nmodes = Nmodes_from_NDeltad(NDeltad);
    Eigen::MatrixXcd ret = Eigen::MatrixXcd::Identity(2*Nmodes, 2*Nmodes);
    const std::complex<double> I(0, 1);
    double last_phase = 0;
    for (int i = 0; i < NAtoms; ++i) {
        const double total_phase = phases[i];
        ret = free_matrix_N(total_phase-last_phase, Nmodes)*ret;
        const std::complex<double> OmegaPhaseFactor = std::exp(2.0*I*total_phase);
        Eigen::MatrixXcd S_r = dual_color_S_r(NDeltad, Deltac, Deltad, delta, g1d, Omega, OmegaPhaseFactor);
        ret = scattering_matrixN_beta(S_r)*ret;
        last_phase = total_phase;
    }
    const double final_total_phase = NAtoms*mean_kd*M_PI;
    const double phase_diff = final_total_phase-last_phase;
    ret = free_matrix_N(phase_diff, Nmodes)*ret;
    return ret;
}

std::vector<double> generate_random_propagation_phases(int NAtoms, double mean_kd,
                                                       int randomSeed,
                                                       bool regularPlacement)
{
    std::mt19937 generator(randomSeed);
    std::vector<double> phases(NAtoms);
    if (regularPlacement) {
        for (int i = 0; i < NAtoms; ++i) {
            phases[i] = static_cast<double>(i)/NAtoms;
        }
    } else {
        generate_random_atom_positions(phases.data(), generator, NAtoms);
    }
    for (int i = 0; i < NAtoms; ++i) {
        phases[i] *= NAtoms*mean_kd*M_PI;
    }
    return phases;
}
} // unnamed namespace

std::complex<double> qd_eit_standing_wave(std::complex<double> delta, double Deltac,
                                          double Omega, double g1d,
                                          double gprime, unsigned periodLength,
                                          double phaseShift)
{
    const double kd = M_PI*(1.0+1.0/periodLength);
    Eigen::Matrix2cd Mcell = Eigen::Matrix2cd::Identity();
    for (int i = 0; i < periodLength; ++i) {
        double OmegaAtThisSite = Omega*std::cos(kd*i+M_PI*phaseShift);
        Eigen::Matrix2cd M3 = scattering_matrix(beta3(delta, Deltac, OmegaAtThisSite, g1d, gprime));
        Eigen::Matrix2cd Mf = free_matrix(kd);
        Mcell = Mf*M3*Mcell;
    }
    std::complex<double> ret = (1.0/periodLength)*std::acos(0.5*Mcell.trace())-double(M_PI/periodLength);
    return ret;
}

QDOnBlochBandRandom::QDOnBlochBandRandom(int classicalDrivePeriods,
                                         LevelScheme level_scheme,
                                         bool regularPlacement) :
    m_aboutToSwitchToAHigherBranch(false),
    m_aboutToSwitchToALowerBranch(false),
    m_regularPlacement(regularPlacement),
    m_deltaLast(-HUGE_VAL),
    m_kdLast(0),
    m_NAtoms(0),
    m_level_scheme(level_scheme),
    m_classicalDrivePeriods(classicalDrivePeriods),
    m_randomSeed(12345),
    m_branchShift(0)
{
}

double QDOnBlochBandRandom::meanKD() const
{
    return static_cast<double>(m_classicalDrivePeriods)/m_NAtoms;
}

#define SORT_BLOCH_VECTORS

#define LAMBDA_BLOCH_KD_DEBUG
std::vector<std::complex<double>> QDOnBlochBandRandom::qdLambda(
        int NAtoms, int NDeltad, std::complex<double> delta,
        double Deltac, double Deltad, double Omega, double g1d,
        double shift, double distributionWidth)
{
    const std::complex<double> I(0,1);
    const double gprime = 1-g1d;
    setRandomSeed(m_randomSeed, NAtoms);
    const double mean_kd = meanKD();
    Eigen::Matrix2cd M
            = M_trans_stationary_light_lambda(m_phases, NAtoms, mean_kd,
                                          delta, Deltac, Omega, g1d, gprime);
    if (delta.real() < m_deltaLast) {
        // Here we rely on the caller to supply
        // the Delta values in the increasing order.
        // If the next Delta value is less than the
        // previous one, then we assume that it is
        // because a new plotting run (possibly with
        // other parameters changed too) was initiated.
        m_branchShift = 0;
    }
    m_deltaLast = delta.real();

    Eigen::ComplexEigenSolver<Eigen::Matrix2cd> es;
    es.compute(M);

    const int Nbloch = 2;
    std::vector<int> i_permutations(Nbloch);
    for (int i = 0; i < Nbloch; ++i) {
        i_permutations[i] = i;
    }
#ifdef SORT_BLOCH_VECTORS
    if (m_last_eigenvector_matrix.cols() != Nbloch || m_last_eigenvector_matrix.rows() != Nbloch) {
        m_last_eigenvector_matrix = Eigen::MatrixXcd::Zero(Nbloch, Nbloch);
    }
    for (int i = 0; i < Nbloch; ++i) {
        int i_permuted = i;
        double max_overlap = 0;
        for (int j = 0; j < Nbloch; ++j) {
            const double overlap = std::norm(es.eigenvectors().col(j).dot(m_last_eigenvector_matrix.col(i)));
            if (overlap > max_overlap) {
                max_overlap = overlap;
                i_permuted = j;
            }
        }
        i_permutations[i] = i_permuted;
        //Check that this permutation has not been taken already
        for (int j = 0; j < i; ++j) {
            if (i_permutations[j] == i_permuted) {
                // Revert to the default value
                i_permutations[i] = i;
                break;
            }
        }
    }
    m_last_eigenvector_matrix = es.eigenvectors();
    for (int i = 0; i < Nbloch; ++i) {
        m_last_eigenvector_matrix.col(i) = es.eigenvectors().col(i_permutations[i]);
    }
#endif // SORT_BLOCH_VECTORS
    // (-1)^{m_classicalDrivePeriods}
    int correction_factor;
    if (m_classicalDrivePeriods % 2 == 1) {
        correction_factor = -1;
    } else {
        correction_factor = 1;
    }

    std::vector<std::complex<double>> kd_array(2);
    for (int i = 0; i < Nbloch; ++i) {
        int i_permuted = i_permutations[i];
        kd_array[i] = -I*std::log(static_cast<double>(correction_factor)
                                  *es.eigenvalues()(i_permuted))
                      /static_cast<double>(NAtoms);
    }
    return kd_array;
}

std::vector<std::complex<double>> QDOnBlochBandRandom::qdDualV(
        int NAtoms, int NDeltad, std::complex<double> delta,
        double Deltac, double Deltad, double Omega, double g1d,
        double shift, double distributionWidth)
{
    const std::complex<double> I(0,1);
    setRandomSeed(m_randomSeed, NAtoms);
    const double mean_kd = meanKD();
    Eigen::Matrix4cd M
            = M_trans_stationary_light_dualv(m_phases, NAtoms, mean_kd,
                                        delta, Deltac, Omega, g1d);

    if (delta.real() < m_deltaLast) {
        // Here we rely on the caller to supply
        // the Delta values in the increasing order.
        // If the next Delta value is less than the
        // previous one, then we assume that it is
        // because a new plotting run (possibly with
        // other parameters changed too) was initiated.
        m_branchShift = 0;
    }
    m_deltaLast = delta.real();

    Eigen::ComplexEigenSolver<Eigen::Matrix4cd> es;
    es.compute(M);

    // (-1)^{m_classicalDrivePeriods}
    int correction_factor;
    if (m_classicalDrivePeriods % 2 == 1) {
        correction_factor = -1;
    } else {
        correction_factor = 1;
    }

    std::vector<std::complex<double>> kd_array(4);

    for (int i = 0; i < 4; ++i) {
        kd_array[i] = -I*std::log(static_cast<double>(correction_factor)
                                  *es.eigenvalues()(i))
                      /static_cast<double>(NAtoms);
    }
    return kd_array;
}


std::vector<std::complex<double>> QDOnBlochBandRandom::qdDualColor(
        int NAtoms, int NDeltad, std::complex<double> delta,
        double Deltac, double Deltad, double Omega, double g1d,
        double shift, double distributionWidth)
{
    const std::complex<double> I(0,1);
    const int Nmodes = Nmodes_from_NDeltad(NDeltad);
    const int Nbloch = 2*Nmodes;
    setRandomSeed(m_randomSeed, NAtoms);
    const double mean_kd = meanKD();
    Eigen::MatrixXcd M
            = M_trans_stationary_light_dual_color(m_phases, NAtoms, NDeltad, mean_kd,
                                             delta, Deltac, Deltad, Omega, g1d);

    if (delta.real() < m_deltaLast) {
        // Here we rely on the caller to supply
        // the Delta values in the increasing order.
        // If the next Delta value is less than the
        // previous one, then we assume that it is
        // because a new plotting run (possibly with
        // other parameters changed too) was initiated.
        m_branchShift = 0;
    }
    m_deltaLast = delta.real();

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es;
    es.compute(M);

    std::vector<int> i_permutations(Nbloch);
    for (int i = 0; i < Nbloch; ++i) {
        i_permutations[i] = i;
    }
#ifdef SORT_BLOCH_VECTORS
    if (m_last_eigenvector_matrix.cols() != Nbloch || m_last_eigenvector_matrix.rows() != Nbloch) {
        m_last_eigenvector_matrix = Eigen::MatrixXcd::Zero(Nbloch, Nbloch);
    }
    for (int i = 0; i < Nbloch; ++i) {
        int i_permuted = i;
        double max_overlap = 0;
        for (int j = 0; j < Nbloch; ++j) {
            const double overlap = std::norm(es.eigenvectors().col(j).dot(m_last_eigenvector_matrix.col(i)));
            if (overlap > max_overlap) {
                max_overlap = overlap;
                i_permuted = j;
            }
        }
        i_permutations[i] = i_permuted;
        //Check that this permutation has not been taken already
        for (int j = 0; j < i; ++j) {
            if (i_permutations[j] == i_permuted) {
                // Revert to the default value
                i_permutations[i] = i;
                break;
            }
        }
    }
    m_last_eigenvector_matrix = es.eigenvectors();
    for (int i = 0; i < Nbloch; ++i) {
        m_last_eigenvector_matrix.col(i) = es.eigenvectors().col(i_permutations[i]);
    }
#endif // SORT_BLOCH_VECTORS

    // (-1)^{m_classicalDrivePeriods}
    int correction_factor;
    if (m_classicalDrivePeriods % 2 == 1) {
        correction_factor = -1;
    } else {
        correction_factor = 1;
    }

    std::vector<std::complex<double>> kd_array(Nbloch);
    for (int i = 0; i < Nbloch; ++i) {
        int i_permuted = i_permutations[i];
        kd_array[i] = -I*std::log(static_cast<double>(correction_factor)
                                  *es.eigenvalues()(i_permuted))
                      /static_cast<double>(NAtoms);
    }
    return kd_array;
}

void QDOnBlochBandRandom::setRandomSeed(int randomSeed, int NAtoms)
{
    if (m_randomSeed == randomSeed && m_NAtoms == NAtoms) {
        return;
    }
    m_NAtoms = NAtoms;
    m_randomSeed = randomSeed;
    const double mean_kd = meanKD();
    m_phases = generate_random_propagation_phases(m_NAtoms, mean_kd,
                                                  m_randomSeed, m_regularPlacement);
}
