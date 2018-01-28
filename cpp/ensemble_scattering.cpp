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

#include "ensemble_scattering.h"

#include <iostream>

#include "Eigen/Dense"
#include "unsupported/Eigen/MatrixFunctions"

#include "dual_v_scattering.h"
#include "hamiltonian_params.h"
#include "urandom.h"

EnsembleScattering::EnsembleScattering(int flags) :
    m_NAtoms(0),
    m_kdEnsemble(0),
    m_kdOmega(0),
    m_delta(0),
    m_Deltac(0),
    m_g1d(0),
    m_Omega(0),
    m_C_6(0),
    m_flags(flags),
    m_numRealizationsToSkip(0),
    m_atomPositions(),
    m_OmegaStrengths(),
    m_phaseFactors()
{
    std::random_device randomDevice;
    m_randomSeed = randomDevice();
}

// The copy constructor is defined explicitely
// to work around an internal compiler error in GCC 6.1
EnsembleScattering::EnsembleScattering(
        const EnsembleScattering &other) :
    m_NAtoms(other.m_NAtoms),
    m_kdEnsemble(other.m_kdEnsemble),
    m_kdOmega(other.m_kdOmega),
    m_delta(other.m_delta),
    m_Deltac(other.m_Deltac),
    m_g1d(other.m_g1d),
    m_Omega(other.m_Omega),
    m_C_6(other.m_C_6),
    m_flags(other.m_flags),
    m_randomSeed(other.m_randomSeed),
    m_numRealizationsToSkip(other.m_numRealizationsToSkip),
    m_atomPositions(other.m_atomPositions),
    m_OmegaStrengths(other.m_OmegaStrengths),
    m_phaseFactors(other.m_phaseFactors),
    m_transferMatrices2(other.m_transferMatrices2),
    m_transferMatrices4(other.m_transferMatrices4)
{
}

void EnsembleScattering::fillAtomArraysNewValues(int NAtoms, double kd_ensemble, double kd_Omega)
{
    if (m_NAtoms != NAtoms || m_kdEnsemble != kd_ensemble
            || m_kdOmega != kd_Omega) {
        m_NAtoms = NAtoms;
        m_kdEnsemble = kd_ensemble;
        m_kdOmega = kd_Omega;
        fillAtomArrays();
        fillTransferMatrixArray();
    }
}

void EnsembleScattering::fillAtomArraysNewValues(
        int NAtoms, double kd_ensemble, double kd_Omega,
        double delta, double Deltac, double g1d, double Omega)
{
    if (m_NAtoms != NAtoms || m_kdEnsemble != kd_ensemble
            || m_kdOmega != kd_Omega) {
        m_NAtoms = NAtoms;
        m_kdEnsemble = kd_ensemble;
        m_kdOmega = kd_Omega;
        fillAtomArrays();
    }
    fillTransferMatrixArrayNewValues(delta, Deltac, g1d, Omega);
}

void EnsembleScattering::fillAtomArrays()
{
    double effectiveKdOmega = m_kdOmega;
    if (effectiveKdOmega <= 0) {
        effectiveKdOmega = m_kdEnsemble;
    }
    double gratingShift = 0;

    m_atomPositions.clear();
    // It is slightly easier for the code below and some of
    // the code that uses this class to include a "fictitious"
    // terminating position in this array which is equal to 1.0.
    // (I.e. the full length of the ensemble in rescaled units.)
    m_atomPositions = std::vector<double>(m_NAtoms+1);
    m_atomPositions[m_NAtoms] = 1.0;

    m_OmegaStrengths.clear();
    m_OmegaStrengths = std::vector<double>(m_NAtoms);

    m_OmegaPhaseFactors.clear();
    m_OmegaPhaseFactors = std::vector<std::complex<double>>(m_NAtoms);

    m_phaseFactors.clear();
    m_phaseFactors = std::vector<std::complex<double>>(m_NAtoms);

    const std::complex<double> I(0,1);
    if (randomAtomPositions()) {
        std::mt19937 generator(m_randomSeed);
        generator.discard(m_NAtoms*m_numRealizationsToSkip);
        generate_random_atom_positions(m_atomPositions.data(), generator,
                                       m_NAtoms);
        for (int i = 0; i < m_NAtoms; ++i) {
            const double totalPhase = m_atomPositions[i]*m_NAtoms*m_kdEnsemble*M_PI;
            m_OmegaStrengths[i] = std::cos(totalPhase);
            m_OmegaPhaseFactors[i] = std::exp(2.0*I*totalPhase);
        }
        double lastTotalPhase = 0;
        for (int i = 0; i < m_NAtoms; ++i) {
            // Note that m_atomPositions[m_NAtoms] = 1.0
            const double totalPhaseNext = m_atomPositions[i+1]*m_NAtoms*m_kdEnsemble*M_PI;
            m_phaseFactors[i] = std::exp(I*(totalPhaseNext-lastTotalPhase));
            lastTotalPhase = totalPhaseNext;
        }
    } else {
        const double gridSpacing = grid_spacing(m_NAtoms, 1);
        const std::complex<double> phaseFactor = std::exp(I*m_kdEnsemble*M_PI);
        for (int i = 0; i < m_NAtoms; ++i) {
            m_atomPositions[i] = i*gridSpacing;
            m_phaseFactors[i] = phaseFactor;
            m_OmegaStrengths[i] = std::cos(effectiveKdOmega*M_PI*double(i)+gratingShift);
            m_OmegaPhaseFactors[i] = std::exp(2.0*I*m_kdEnsemble*M_PI*static_cast<double>(i));
        }
    }
}
void EnsembleScattering::fillTransferMatrixArrayNewValues(
        double delta, double Deltac, double g1d, double Omega)
{
    if (m_delta != delta || m_Deltac != Deltac || m_g1d != g1d || m_Omega != Omega) {
        m_delta = delta;
        m_Deltac = Deltac;
        m_g1d = g1d;
        m_Omega = Omega;
        fillTransferMatrixArray();
    }
}

void EnsembleScattering::fillTransferMatrixArray()
{
    if (m_flags & ENSEMBLE_SCATTERING_DUAL_V_ATOMS) {
        fillTransferMatrixArrayDualV();
    }
}

inline void r_and_t_from_4by4_transfer_matrix(std::complex<double> &rpm,
                                              std::complex<double> &tpp,
                                              const Eigen::Matrix4cd &M)
{
    Eigen::Vector2cd E_R_zm;
    // There is only a \sigma^+ field incident from
    // the left:
    E_R_zm(0) = 1;
    E_R_zm(1) = 0;

    Eigen::Matrix2cd T_00 = M.block<2,2>(0,0);
    Eigen::Matrix2cd T_01 = M.block<2,2>(0,2);
    Eigen::Matrix2cd T_10 = M.block<2,2>(2,0);
    Eigen::Matrix2cd T_11 = M.block<2,2>(2,2);
    Eigen::FullPivLU<Eigen::Matrix2cd> lu(T_11);

    Eigen::Vector2cd E_L_zm = -lu.solve(T_10*E_R_zm);
    Eigen::Vector2cd E_R_zp = (T_00-T_01*lu.solve(T_10))*E_R_zm;
    tpp = E_R_zp(0);
    rpm = E_L_zm(1);
}

void EnsembleScattering::fillTransferMatrixArrayDualV()
{
    if (m_transferMatrices4.size() != m_NAtoms) {
        m_transferMatrices4 = std::vector<Eigen::Matrix4cd, Eigen::aligned_allocator<Eigen::Matrix4cd>>(m_NAtoms);
    }
    #pragma omp parallel for
    for (int i = 0; i < m_NAtoms; ++i) {
        std::complex<double> rpp;
        std::complex<double> tpp;
        std::complex<double> rpm;
        std::complex<double> tpm;
        std::complex<double> rmm;
        std::complex<double> tmm;
        std::complex<double> rmp;
        std::complex<double> tmp;
        dualv_scattering_coefficients(rpp, tpp, rmp, tmp, rpm, tpm, rmm, tmm,
                                      m_Deltac, m_Deltac, m_delta, m_g1d,
                                      m_Omega, m_Omega,
                                      m_OmegaPhaseFactors[i]);
        Eigen::Matrix4cd Matom = scattering_matrix4_beta(rpp, rmp, rpm, rmm);
        Eigen::Matrix4cd MfToNextAtom = Eigen::Matrix4cd::Zero();
        MfToNextAtom(0,0) = m_phaseFactors[i];
        MfToNextAtom(1,1) = m_phaseFactors[i];
        MfToNextAtom(2,2) = std::conj(m_phaseFactors[i]);
        MfToNextAtom(3,3) = std::conj(m_phaseFactors[i]);
        m_transferMatrices4[i] = MfToNextAtom*Matom;
    }
}

void EnsembleScattering::setC_6(double C_6)
{
    m_C_6 = C_6;
}

void EnsembleScattering::setRandomSeed(unsigned long int seed, unsigned long numRealizationsToSkip)
{
    if ((m_randomSeed == seed
         && m_numRealizationsToSkip == numRealizationsToSkip)
         || !randomAtomPositions()) {
        return;
    }
    m_randomSeed = seed;
    m_numRealizationsToSkip = numRealizationsToSkip;
    if (m_NAtoms == 0) {
        return;
    }
    fillAtomArrays();
    fillTransferMatrixArray();
}

namespace {
inline void
addAtomAtPosition(int i, double delta, double Deltac, double g1d, double Omega,
                  const std::vector<double> &OmegaStrengths,
                  const std::vector<std::complex<double>> &phaseFactors,
                  bool randomAtomPositions, const Eigen::Matrix2cd &Mf,
                  Eigen::Matrix2cd &MLeft)
{
    const std::complex<double> I(0,1);
    const double OmegaAtThisSite = Omega*OmegaStrengths[i];
    std::complex<double> betaAtThisSite = 0;
    const double Delta = delta+Deltac;
    if (std::abs(OmegaStrengths[i]) > 1e-16) {
        // Only if the |cos(kz)| is bigger than the
        // machine epsilon that we use the EIT formula
        betaAtThisSite = g1d*delta
                         /(((1-g1d)-2.0*I*Delta)*delta
                            +2.0*I*std::pow(OmegaAtThisSite, 2));
    } else {
        // Otherwise, 1e-16 is essentially equal to zero
        // in the double precision arithmetic. Hence, this
        // atom is effectively a two-level one.
        betaAtThisSite = g1d/(((1-g1d)-2.0*I*Delta));
    }
    Eigen::Matrix2cd Mbeta;
    Mbeta(0,0) = 1.0-betaAtThisSite;
    Mbeta(0,1) = -betaAtThisSite;
    Mbeta(1,0) = betaAtThisSite;
    Mbeta(1,1) = 1.0+betaAtThisSite;
    Eigen::Matrix2cd MfToNextAtom = Mf;
    if (randomAtomPositions) {
        MfToNextAtom(0,0) = phaseFactors[i];
        MfToNextAtom(0,1) = 0;
        MfToNextAtom(1,0) = 0;
        MfToNextAtom(1,1) = std::conj(phaseFactors[i]);
    }
    MLeft = MfToNextAtom*Mbeta*MLeft;
}

inline void
addDualVAtomAtPosition(int i, double delta, double Deltac, double g1d, double Omega,
                  const std::vector<std::complex<double>> &OmegaPhaseFactors,
                  const std::vector<std::complex<double>> &phaseFactors,
                  const std::vector<double> &atomPositions,
                  const double kd_ensemble,
                  bool randomAtomPositions, const Eigen::Matrix4cd &Mf,
                  Eigen::Matrix4cd &MArray)
{
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
                                  Omega, Omega, OmegaPhaseFactors[i]);
    Eigen::Matrix4cd Matom = scattering_matrix4_beta(rpp, rmp, rpm, rmm);
    Eigen::Matrix4cd MfToNextAtom = Mf;
    if (randomAtomPositions) {
        MfToNextAtom(0,0) = phaseFactors[i];
        MfToNextAtom(1,1) = phaseFactors[i];
        MfToNextAtom(2,2) = std::conj(phaseFactors[i]);
        MfToNextAtom(3,3) = std::conj(phaseFactors[i]);
    }
    MArray = MfToNextAtom*Matom*MArray;
}
} // unnamed namespace

RandTCoefficients
EnsembleScattering::operator()(
        double delta, double kd_ensemble, double kd_Omega,
        double Deltac, double g1d, double Omega, int NAtoms,
        int impurityPosition)
{
    if (m_flags & ENSEMBLE_SCATTERING_DUAL_V_ATOMS) {
        if (m_flags & ENSEMBLE_SCATTERING_RYDBERG) {
            return rAndTCoefficientsDualVRydberg(delta, kd_ensemble, kd_Omega,
                                                 Deltac, g1d, Omega, NAtoms,
                                                 impurityPosition);
        } else {
            return rAndTCoefficientsDualV(delta, kd_ensemble, kd_Omega,
                                          Deltac, g1d, Omega, NAtoms,
                                          impurityPosition);
        }
    } else {
        if (m_flags & ENSEMBLE_SCATTERING_RYDBERG) {
            return rAndTCoefficientsLambdaRydberg(delta, kd_ensemble, kd_Omega,
                                                  Deltac, g1d, Omega, NAtoms,
                                                  impurityPosition);
        } else {
            return rAndTCoefficientsLambda(delta, kd_ensemble, kd_Omega,
                                           Deltac, g1d, Omega, NAtoms,
                                           impurityPosition);
        }
    }
}

RandTCoefficients
EnsembleScattering::rAndTCoefficientsLambda(
        double delta, double kd_ensemble, double kd_Omega,
        double Deltac, double g1d, double Omega, int NAtoms,
        int impurityPosition)
{
    assert(impurityPosition < NAtoms && "Too big impurity position!");
    assert(impurityPosition >= 0 && "Impurity position is negative!");

    fillAtomArraysNewValues(NAtoms, kd_ensemble, kd_Omega);
    const std::complex<double> I(0,1);
    const std::complex<double> beta2_imp = g1d/((1-g1d));
    Eigen::Matrix2cd Mf;
    Mf(0,0) = std::exp(I*kd_ensemble*M_PI);
    Mf(0,1) = 0;
    Mf(1,0) = 0;
    Mf(1,1) = std::exp(-I*kd_ensemble*M_PI);
    Eigen::Matrix2cd MLeft = Eigen::Matrix2cd::Identity();
    Eigen::Matrix2cd MRight = Eigen::Matrix2cd::Identity();

    // Optimization for a repeating unit cell:
    // compute the unit cell here and then exponentiate
    // it in the following where appropriate.
    const int periodLength = find_period_length_from_kd(kd_ensemble);
    const bool rap = randomAtomPositions();
    bool enableUnitCellExponentiating;
    if (periodLength > 0 && !rap) {
        enableUnitCellExponentiating = true;
    } else {
        enableUnitCellExponentiating = false;
    }

    Eigen::Matrix2cd MUnitCell = Eigen::Matrix2cd::Identity();
    if (enableUnitCellExponentiating) {
        for (int i = 0; i < periodLength; ++i) {
            addAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaStrengths,
                              m_phaseFactors, false, Mf, MUnitCell);
        }
    }

    int i = 0;
    if (enableUnitCellExponentiating) {
        const int numUnitCells = impurityPosition / periodLength;
        Eigen::MatrixPower<Eigen::Matrix2cd> MUnitCellPow(MUnitCell);
        MLeft = MUnitCellPow(numUnitCells)*MLeft;
        i += numUnitCells*periodLength;
    }
    for (; i < impurityPosition; ++i) {
        addAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaStrengths,
                          m_phaseFactors, rap, Mf, MLeft);
    }
    Eigen::Matrix2cd Matom = Eigen::Matrix2cd::Identity();
    Eigen::Matrix2cd M_impurity_atom = Eigen::Matrix2cd::Identity();
    Eigen::Matrix2cd M2_cd;
    M2_cd(0,0) = 1.0-beta2_imp;
    M2_cd(0,1) = -beta2_imp;
    M2_cd(1,0) = beta2_imp;
    M2_cd(1,1) = 1.0+beta2_imp;
    {
        addAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaStrengths,
                          m_phaseFactors, rap, Mf, Matom);
        Eigen::Matrix2cd MfToNextAtom = Mf;
        if (rap) {
            MfToNextAtom(0,0) = m_phaseFactors[i];
            MfToNextAtom(0,1) = 0;
            MfToNextAtom(1,0) = 0;
            MfToNextAtom(1,1) = std::conj(m_phaseFactors[i]);
        }
        M_impurity_atom = MfToNextAtom*M2_cd*M_impurity_atom;
        ++i;
    }
    int NAtomsToManuallySum = m_NAtoms;
    if (enableUnitCellExponentiating) {
        const int currentUnitCell = i/periodLength + 1;
        NAtomsToManuallySum = std::min(currentUnitCell*periodLength, m_NAtoms);
    }
    for (; i < NAtomsToManuallySum; ++i) {
        addAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaStrengths,
                          m_phaseFactors, rap, Mf, MRight);
    }
    if (enableUnitCellExponentiating) {
        const int numUnitCells = (m_NAtoms-NAtomsToManuallySum) / periodLength;
        Eigen::MatrixPower<Eigen::Matrix2cd> MUnitCellPow(MUnitCell);
        MRight = MUnitCellPow(numUnitCells)*MRight;
        i += numUnitCells*periodLength;
        // Multiply the remaining atoms
        // that didn't comprise a whole unit cell
        for (; i < m_NAtoms; ++i) {
            addAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaStrengths,
                              m_phaseFactors, rap, Mf, MRight);
        }
    }

    Eigen::Matrix2cd Mfull;
    Eigen::Matrix2cd Mfull_impurity;
    Mfull = MRight*Matom*MLeft;
    Mfull_impurity = MRight*M_impurity_atom*MLeft;
    RandTCoefficients ret;
    ret.Mensemble = Mfull;
    ret.Mensemble_impurity = Mfull_impurity;
    // Reflection coefficients are taken to be the ones
    // from the left. (There should not be any big difference
    // between scattering from the left and from the right though.)
    ret.t = 1.0/Mfull(1,1);
    ret.t_impurity = 1.0/Mfull_impurity(1,1);
#ifdef REFLECTION_COEFFICIENT_SCATTERING_FROM_THE_LEFT
    ret.r = -Mfull(1,0)/Mfull(1,1);
    ret.r_impurity = -Mfull_impurity(1,0)/Mfull_impurity(1,1);
#else // REFLECTION_COEFFICIENT_SCATTERING_FROM_THE_LEFT
    // Here we return the reflection coefficients for scattering
    // from the right. Scattering from the left and from the right
    // should be essentially the same (at least in the limit of
    // many atoms).
    ret.r = Mfull(0,1)/Mfull(1,1);
    ret.r_impurity = Mfull_impurity(0,1)/Mfull_impurity(1,1);
#endif // REFLECTION_COEFFICIENT_SCATTERING_FROM_THE_LEFT
    return ret;
}

RandTCoefficients
EnsembleScattering::rAndTCoefficientsDualV(
        double delta, double kd_ensemble, double kd_Omega,
        double Deltac, double g1d, double Omega, int NAtoms,
        int impurityPosition)
{
    assert(impurityPosition < NAtoms && "Too big impurity position!");
    assert(impurityPosition >= 0 && "Impurity position is negative!");

    const bool rap = randomAtomPositions();
    // This is an optimization for the case
    // when this function is called many times
    // with the same delta, Deltac, g1d and Omega
    // and different impurityPosition. This happens
    // for example during the calculation of the
    // fidelity.
    fillAtomArraysNewValues(NAtoms, kd_ensemble, kd_Omega, delta, Deltac, g1d, Omega);
    const std::complex<double> I(0,1);

    Eigen::Matrix4cd Mf = Eigen::Matrix4cd::Zero();
    Mf(0,0) = std::exp(I*kd_ensemble*M_PI);
    Mf(1,1) = std::exp(I*kd_ensemble*M_PI);
    Mf(2,2) = std::exp(-I*kd_ensemble*M_PI);
    Mf(3,3) = std::exp(-I*kd_ensemble*M_PI);

    Eigen::Matrix4cd MLeft = Eigen::Matrix4cd::Identity();
    Eigen::Matrix4cd MRight = Eigen::Matrix4cd::Identity();

    int i = 0;
    for (; i < impurityPosition; ++i) {
        MLeft = m_transferMatrices4[i]*MLeft;
    }

    Eigen::Matrix4cd Matom = Eigen::Matrix4cd::Identity();
    Eigen::Matrix4cd M_impurity_atom = Eigen::Matrix4cd::Identity();
    {
        addDualVAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaPhaseFactors,
                          m_phaseFactors, m_atomPositions, m_kdEnsemble, rap,
                          Mf, Matom);
        const std::complex<double> rpp = -g1d;
        const std::complex<double> rpm = 0;
        const std::complex<double> rmm = rpp;
        const std::complex<double> rmp = 0;
        Eigen::Matrix4cd M_imp = scattering_matrix4_beta(rpp, rmp, rpm, rmm);
        Eigen::Matrix4cd MfToNextAtom = Mf;
        if (rap) {
            MfToNextAtom(0,0) = m_phaseFactors[i];
            MfToNextAtom(1,1) = m_phaseFactors[i];
            MfToNextAtom(2,2) = std::conj(m_phaseFactors[i]);
            MfToNextAtom(3,3) = std::conj(m_phaseFactors[i]);
        }
        M_impurity_atom = MfToNextAtom*M_imp*M_impurity_atom;
        ++i;
    }
    for (; i < m_NAtoms; ++i) {
        MRight = m_transferMatrices4[i]*MRight;
    }

    Eigen::Matrix4cd Mfull = MRight*Matom*MLeft;
    Eigen::Matrix4cd Mfull_impurity = MRight*M_impurity_atom*MLeft;

    RandTCoefficients ret;
    ret.Mensemble = Mfull;
    ret.Mensemble_impurity = Mfull_impurity;
    r_and_t_from_4by4_transfer_matrix(ret.r, ret.t, Mfull);
    r_and_t_from_4by4_transfer_matrix(ret.r_impurity, ret.t_impurity, Mfull_impurity);
    return ret;
}

namespace {
inline double rydberg_potential(double z, double z_imp, double C_6)
{
    return C_6/std::pow(z-z_imp,6);
}

inline double rydberg_potential_square(double z, double z_imp, double C_6)
{
    const double z_b = std::pow(std::abs(C_6), 1.0/6);
    if (z > z_imp-z_b && z < z_imp+z_b) {
        return 1;
    } else {
        return 0;
    }
}
}

RandTCoefficients
EnsembleScattering::rAndTCoefficientsLambdaRydberg(
        double delta, double kd_ensemble, double kd_Omega,
        double Deltac, double g1d, double Omega, int NAtoms,
        int impurityPosition)
{
    assert(impurityPosition < NAtoms && "Too big impurity position!");
    assert(impurityPosition >= 0 && "Impurity position is negative!");

    fillAtomArraysNewValues(NAtoms, kd_ensemble, kd_Omega);
    const std::complex<double> I(0,1);
    Eigen::Matrix2cd Mf;
    Mf(0,0) = std::exp(I*kd_ensemble*M_PI);
    Mf(0,1) = 0;
    Mf(1,0) = 0;
    Mf(1,1) = std::exp(-I*kd_ensemble*M_PI);
    Eigen::Matrix2cd Mfull = Eigen::Matrix2cd::Identity();
    Eigen::Matrix2cd Mfull_impurity = Eigen::Matrix2cd::Identity();

    const double z_imp = m_atomPositions[impurityPosition];
    const bool rap = randomAtomPositions();
    int i = 0;
    for (; i < impurityPosition; ++i) {
        const double delta_shift = rydberg_potential(m_atomPositions[i], z_imp, m_C_6);
        addAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaStrengths,
                          m_phaseFactors, rap, Mf, Mfull);
        addAtomAtPosition(i, delta+delta_shift, Deltac, g1d, Omega, m_OmegaStrengths,
                          m_phaseFactors, rap, Mf, Mfull_impurity);
    }
    {
        addAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaStrengths,
                          m_phaseFactors, rap, Mf, Mfull);
        Eigen::Matrix2cd MfToNextAtom = Mf;
        if (rap) {
            MfToNextAtom(0,0) = m_phaseFactors[i];
            MfToNextAtom(0,1) = 0;
            MfToNextAtom(1,0) = 0;
            MfToNextAtom(1,1) = std::conj(m_phaseFactors[i]);
        }
        Mfull_impurity = MfToNextAtom*Mfull_impurity;
        ++i;
    }
    for (; i < m_NAtoms; ++i) {
        const double delta_shift = rydberg_potential(m_atomPositions[i], z_imp, m_C_6);
        addAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaStrengths,
                          m_phaseFactors, rap, Mf, Mfull);
        addAtomAtPosition(i, delta+delta_shift, Deltac, g1d, Omega, m_OmegaStrengths,
                          m_phaseFactors, rap, Mf, Mfull_impurity);
    }

    RandTCoefficients ret;
    ret.Mensemble = Mfull;
    ret.Mensemble_impurity = Mfull_impurity;
    // Reflection coefficients are taken to be the ones
    // from the left. (There should not be any big difference
    // between scattering from the left and from the right though.)
    ret.t = 1.0/Mfull(1,1);
    ret.t_impurity = 1.0/Mfull_impurity(1,1);
#ifdef REFLECTION_COEFFICIENT_SCATTERING_FROM_THE_LEFT
    ret.r = -Mfull(1,0)/Mfull(1,1);
    ret.r_impurity = -Mfull_impurity(1,0)/Mfull_impurity(1,1);
#else // REFLECTION_COEFFICIENT_SCATTERING_FROM_THE_LEFT
    // Here we return the reflection coefficients for scattering
    // from the right. Scattering from the left and from the right
    // should be essentially the same (at least in the limit of
    // many atoms).
    ret.r = Mfull(0,1)/Mfull(1,1);
    ret.r_impurity = Mfull_impurity(0,1)/Mfull_impurity(1,1);
#endif // REFLECTION_COEFFICIENT_SCATTERING_FROM_THE_LEFT
    return ret;
}


RandTCoefficients
EnsembleScattering::rAndTCoefficientsDualVRydberg(
        double delta, double kd_ensemble, double kd_Omega,
        double Deltac, double g1d, double Omega, int NAtoms,
        int impurityPosition)
{
    assert(impurityPosition < NAtoms && "Too big impurity position!");
    assert(impurityPosition >= 0 && "Impurity position is negative!");

    const bool rap = randomAtomPositions();
    fillAtomArraysNewValues(NAtoms, kd_ensemble, kd_Omega);
    const std::complex<double> I(0,1);
    const int impurityNumber = static_cast<int>(NAtoms*std::pow(std::abs(m_C_6), 1.0/6));

    Eigen::Matrix4cd Mf = Eigen::Matrix4cd::Zero();
    Mf(0,0) = std::exp(I*kd_ensemble*M_PI);
    Mf(1,1) = std::exp(I*kd_ensemble*M_PI);
    Mf(2,2) = std::exp(-I*kd_ensemble*M_PI);
    Mf(3,3) = std::exp(-I*kd_ensemble*M_PI);

    Eigen::Matrix4cd Mfull = Eigen::Matrix4cd::Identity();
    Eigen::Matrix4cd Mfull_impurity = Eigen::Matrix4cd::Identity();

    const double z_imp = m_atomPositions[impurityPosition];
    int i = 0;
    for (; i < impurityPosition; ++i) {
        const double delta_shift = rydberg_potential(m_atomPositions[i], z_imp, m_C_6);
        addDualVAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaPhaseFactors,
                          m_phaseFactors, m_atomPositions, m_kdEnsemble, rap,
                          Mf, Mfull);
        addDualVAtomAtPosition(i, delta+delta_shift, Deltac, g1d, Omega, m_OmegaPhaseFactors,
                          m_phaseFactors, m_atomPositions, m_kdEnsemble, rap,
                          Mf, Mfull_impurity);
    }
    {
        addDualVAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaPhaseFactors,
                          m_phaseFactors, m_atomPositions, m_kdEnsemble, rap,
                          Mf, Mfull);
        Eigen::Matrix4cd MfToNextAtom = Mf;
        if (rap) {
            MfToNextAtom(0,0) = m_phaseFactors[i];
            MfToNextAtom(1,1) = m_phaseFactors[i];
            MfToNextAtom(2,2) = std::conj(m_phaseFactors[i]);
            MfToNextAtom(3,3) = std::conj(m_phaseFactors[i]);
        }
        Mfull_impurity = MfToNextAtom*Mfull_impurity;
        ++i;
    }
    for (; i < m_NAtoms; ++i) {
        const double delta_shift = rydberg_potential(m_atomPositions[i], z_imp, m_C_6);
        addDualVAtomAtPosition(i, delta, Deltac, g1d, Omega, m_OmegaPhaseFactors,
                          m_phaseFactors, m_atomPositions, m_kdEnsemble, rap,
                          Mf, Mfull);
        addDualVAtomAtPosition(i, delta+delta_shift, Deltac, g1d, Omega, m_OmegaPhaseFactors,
                          m_phaseFactors, m_atomPositions, m_kdEnsemble, rap,
                          Mf, Mfull_impurity);
    }

    RandTCoefficients ret;
    ret.Mensemble = Mfull;
    ret.Mensemble_impurity = Mfull_impurity;
    r_and_t_from_4by4_transfer_matrix(ret.r, ret.t, Mfull);
    r_and_t_from_4by4_transfer_matrix(ret.r_impurity, ret.t_impurity, Mfull_impurity);
    return ret;
}
