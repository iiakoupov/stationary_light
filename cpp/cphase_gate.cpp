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

#include "cphase_gate.h"
#include "adiabatic_eit.h"
#include "ensemble_scattering.h"
#include "gaussian_modes.h"
#include "gaussian_electric_field_modes.h"
#include "urandom.h"
#include "gate_common.h"
#include "findroot.h"

#include <nlopt.hpp>

#include <iostream>
#include <fstream>

//#define CPHASE_GATE_DEBUG_PRINT

inline void r_and_t_from_both_sides(std::complex<double> &r_plus, std::complex<double> &r_minus,
                                    std::complex<double> &t_plus, std::complex<double> &t_minus,
                                    const Eigen::MatrixXcd &M)
{
    // In Sagnac scattering the field is incident
    // from both sides of the ensemble.
    // We explicitely compute the reflection and
    // transmission coefficients for the whole
    // ensemble for the field, that is incident
    // from the left of the ensemble
    // (r_plus and t_plus)
    // and from the right of the ensemble
    // (r_minus and t_minus).
    if (M.rows() == 2) {
        // 2 by 2 transfer matrices
        r_plus = -M(1,0)/M(1,1);
        r_minus = M(0,1)/M(1,1);
        t_plus = 1.0/M(1,1);
        t_minus = t_plus;
    } else if (M.rows() == 4) {
        // 4 by 4 transfer matrices

        // First we calculate with incident
        // field from the left

        // E_+(0^-)
        Eigen::Vector2cd E_R_0m;
        // There is only a \sigma^+ field incident from
        // the left:
        E_R_0m(0) = 1;
        E_R_0m(1) = 0;
        // and E_-(L^+) is a zero vector

        Eigen::Matrix2cd T_00 = M.block<2,2>(0,0);
        Eigen::Matrix2cd T_01 = M.block<2,2>(0,2);
        Eigen::Matrix2cd T_10 = M.block<2,2>(2,0);
        Eigen::Matrix2cd T_11 = M.block<2,2>(2,2);
        Eigen::FullPivLU<Eigen::Matrix2cd> lu(T_11);

        // E_-(0^-), the reflected field in this case
        Eigen::Vector2cd E_L_0m = -lu.solve(T_10*E_R_0m);
        // E_+(L^+), the transmitted field in this case
        Eigen::Vector2cd E_R_Lp = (T_00-T_01*lu.solve(T_10))*E_R_0m;
        // Pick out the right polarization components:
        // for the transmitted field that propagates
        // to the right, it is \sigma^+
        t_plus = E_R_Lp(0);
        // for the reflected field that propagates
        // the the left, it is \sigma^-
        r_plus = E_L_0m(1);

        // Now we calculate with incident
        // field from the right

        // E_-(L^+)
        Eigen::Vector2cd E_L_Lp;
        // There is only a \sigma^- field incident from
        // the right:
        E_L_Lp(0) = 0;
        E_L_Lp(1) = 1;
        // and E_+(0^-) is a zero vector


        // E_-(0^-), the transmitted field in this case
        E_L_0m = lu.solve(E_L_Lp);
        // E_+(L^+), the reflected field in this case
        E_R_Lp = T_01*E_L_0m;

        r_minus = E_R_Lp(0);
        t_minus = E_L_0m(1);
    } else {
        assert(0 && "Unsupported transfer matrix size!");
    }
}

std::complex<double> impurity_reflection_coefficient_discrete(
        int n, double delta, double g1d, int NAtoms, double Deltac, double Omega, double kd,
        double kL1, double kL2,
        EnsembleScattering *ensemble_scattering, int cphaseGateFlags)
{
    RandTCoefficients rAndT = (*ensemble_scattering)(delta, kd, kd, Deltac, g1d, Omega, NAtoms, n);
    // We don't want the propagation phase through the
    // ensemble in the transmission coefficient, so we divide
    // it out. It's equivalent with adding an appropriate
    // length of the free propagation after the ensemble.
    const std::complex<double> I(0,1);
    const std::complex<double> blochPropagationPhaseFactor = std::exp(I*(ensemble_scattering->kdEnsemble()*NAtoms+1)*M_PI);
    const std::complex<double> blochPropagationPhaseFactorSquared = blochPropagationPhaseFactor*blochPropagationPhaseFactor;
    if (cphaseGateFlags & CPHASE_GATE_SAGNAC_SCATTERING) {
        std::complex<double> r_1_plus;
        std::complex<double> t_1_plus;
        std::complex<double> r_1_minus;
        std::complex<double> t_1_minus;
        r_and_t_from_both_sides(r_1_plus, r_1_minus, t_1_plus, t_1_minus, rAndT.Mensemble_impurity);

        // Below, we effectively multiply a matrix of free
        // propagation onto the returned transfer matrix.
        // Suppose we have a 2x2 transfer matrix
        //                            / M_{00}   M_{01} \
        // rAndT.Mensemble_impurity = |                 |
        //                            \ M_{10}   M_{11} /
        //
        // then we multiply the matrix
        //       / e^{ikz}    0       \
        // M_f = |                    |
        //       \    0      e^{-ikz} /
        // where e^{ikz} = 1/blochPropagationPhaseFactor
        // Therefore, we obtain a new matrix
        //                               / M_{00}e^{ikz}   M_{01}e^{ikz}   \
        // Mf*rAndT.Mensemble_impurity = |                                 |
        //                               \ M_{10}e^{-ikz}  M_{11}e^{-ikz}  /
        // Observe that when we compute
        // the reflection coefficients from this adjusted matrix,
        // there is a difference in phases for the reflection
        // coefficients. For incident field from the left, we
        // have the reflection coefficient
        // r_1_plus = -(M_{10}e^{-ikz})/(M_{11}e^{-ikz})=-M_{10}/M_{11},
        // i.e. the extra added phase plays no role, since it was added
        // from the "other" side in this case.
        // On the other hand, for the incident field from the right,
        // we have
        // r_1_minus = (M_{01}e^{ikz})/(M_{11}e^{-ikz}) = (M_{01}/M_{11})e^{2ikz}.
        // In this case the field propagates twice through the extra free space
        // part and therefore acquires twice the phase of free propagation.
        // The transmission coefficients are given by
        // t_1_plus=t_1_minus = 1.0/(M_{11}e^{-ikz}) = (1.0/M_{11})e^{ikz}
        // I.e. they acquire a phase of free propagation (but not twice that
        // as it is for the reflection coefficient).
        //
        // We see also that for 4x4 transfer matrices, a similar
        // argument gives the same end result.

        r_1_minus /= blochPropagationPhaseFactorSquared;
        const std::complex<double> R_1 = -0.5*(r_1_plus*std::exp(2.0*I*kL1*M_PI)+r_1_minus*std::exp(2.0*I*kL2*M_PI) - (t_1_plus+t_1_minus)*std::exp(I*(kL1+kL2)*M_PI)/blochPropagationPhaseFactor);

        // The commented out version below does not take into account
        // the fact that the field is incident from both sides, if the
        // ensemble is placed inside a Sagnac interferometer, and therefore
        // the scattering coefficient can be (slightly) different, depending
        // on the side, from which the field is incident.
        //
        //const std::complex<double> R_1 = -0.5*(rAndT.r_impurity*(std::exp(2.0*I*kL1*M_PI)+std::exp(2.0*I*kL2*M_PI)) - 2.0*rAndT.t_impurity*std::exp(I*(kL1+kL2)*M_PI)/blochPropagationPhaseFactor);

        return R_1;
    } else {
        std::complex<double> r_1_plus;
        std::complex<double> t_1_plus;
        std::complex<double> r_1_minus;
        std::complex<double> t_1_minus;
        r_and_t_from_both_sides(r_1_plus, r_1_minus, t_1_plus, t_1_minus, rAndT.Mensemble_impurity);

        r_1_minus /= blochPropagationPhaseFactorSquared;

        // TODO: does not account for non-zero kL1 and kL2.
        return -(r_1_plus - (t_1_plus*t_1_minus/blochPropagationPhaseFactorSquared)/(1.0+r_1_minus));

        //return -(rAndT.r_impurity - std::pow(rAndT.t_impurity/blochPropagationPhaseFactor, 2)/(1.0+rAndT.r_impurity));
    }
}

std::complex<double> no_impurity_reflection_coefficient(
        double delta, double g1d, int NAtoms, double Deltac, double Omega, double kd,
        double kL1, double kL2,
        EnsembleScattering *ensemble_scattering, int cphaseGateFlags)
{
    RandTCoefficients rAndT = (*ensemble_scattering)(delta, kd, kd, Deltac, g1d, Omega, NAtoms, 0);
    // We don't want the propagation phase through the
    // ensemble in the transmission coefficient, so we divide
    // it out. It's equivalent with adding an appropriate
    // length of the free propagation after the ensemble.
    const std::complex<double> I(0,1);
    const std::complex<double> blochPropagationPhaseFactor = std::exp(I*(ensemble_scattering->kdEnsemble()*NAtoms+1)*M_PI);
    const std::complex<double> blochPropagationPhaseFactorSquared = blochPropagationPhaseFactor*blochPropagationPhaseFactor;
    if (cphaseGateFlags & CPHASE_GATE_SAGNAC_SCATTERING) {
        std::complex<double> r_0_plus;
        std::complex<double> t_0_plus;
        std::complex<double> r_0_minus;
        std::complex<double> t_0_minus;
        r_and_t_from_both_sides(r_0_plus, r_0_minus, t_0_plus, t_0_minus, rAndT.Mensemble);

        // See comment in 'impurity_reflection_coefficient_discrete'
        // (the function above) for explanation of the phase adjustments below.
        r_0_minus /= blochPropagationPhaseFactorSquared;


        const std::complex<double> R_0 = -0.5*(r_0_plus*std::exp(2.0*I*kL1*M_PI)+r_0_minus*std::exp(2.0*I*kL2*M_PI) - (t_0_plus+t_0_minus)*std::exp(I*(kL1+kL2)*M_PI)/blochPropagationPhaseFactor);

        // The commented out version below does not take into account
        // the fact that the field is incident from both sides, if the
        // ensemble is placed inside a Sagnac interferometer, and therefore
        // the scattering coefficient can be (slightly) different, depending
        // on the side, from which the field is incident.
        //
        //const std::complex<double> R_0 = -0.5*(rAndT.r*(std::exp(2.0*I*kL1*M_PI)+std::exp(2.0*I*kL2*M_PI)) - 2.0*rAndT.t*std::exp(I*(kL1+kL2)*M_PI)/blochPropagationPhaseFactor);

        return R_0;
    } else {
        std::complex<double> r_0_plus;
        std::complex<double> t_0_plus;
        std::complex<double> r_0_minus;
        std::complex<double> t_0_minus;
        r_and_t_from_both_sides(r_0_plus, r_0_minus, t_0_plus, t_0_minus, rAndT.Mensemble);

        r_0_minus /= blochPropagationPhaseFactorSquared;

        return -(r_0_plus - (t_0_plus*t_0_minus/blochPropagationPhaseFactorSquared)/(1.0+r_0_minus));

        //return -(rAndT.r - std::pow(rAndT.t/blochPropagationPhaseFactor, 2)/(1.0+rAndT.r));
    }
}

void calculate_cphase_fidelities_spinwave(CphaseFidelities *fid,
                                          const SpinwaveAndFieldVector &sol,
                                          const std::vector<std::complex<double>> &noImpR,
                                          std::complex<double> tNoInteraction)
{
    const int numRealizations = sol.size();
    assert(numRealizations > 0 && "no realizations in the argument!");
    if (numRealizations > 1) {
        std::cout << "Calling this function with more than one realization doesn't make sense!" << std::endl;
    }
    assert(noImpR.size() == numRealizations && "noImpR.size() != fid.size()");
    fid->P_success = 0;
    double F_swap_times_P_success = 0;
    fid->F_CJ = 0;
    for (int i = 0; i < numRealizations; ++i) {
        const double psi_after_storage_squaredNorm = sol[i].psi_after_storage0.squaredNorm();
        fid->single_photon_storage_retrieval_eff = psi_after_storage_squaredNorm;
        const double psi_after_storage_norm = std::sqrt(psi_after_storage_squaredNorm);
        const double psi_after_scattering_squaredNorm = sol[i].psi_after_scattering.squaredNorm();
        fid->P_success
                += 0.25*((2.0*std::norm(tNoInteraction)+std::norm(noImpR[i]))
                        *psi_after_storage_squaredNorm
                        +psi_after_scattering_squaredNorm);
        F_swap_times_P_success
                += (sol[i].psi_after_storage0*(2.0*tNoInteraction+noImpR[i])
                       -sol[i].psi_after_scattering).squaredNorm()/16;
        fid->F_CJ += std::norm((2.0*tNoInteraction+noImpR[i])*psi_after_storage_norm
                              -sol[i].psi_after_storage0.dot(sol[i].psi_after_scattering)/psi_after_storage_norm)/16;
    }
    fid->P_success /= numRealizations;
    F_swap_times_P_success /= numRealizations;
    fid->F_CJ /= numRealizations;
    fid->F_CJ_conditional = fid->F_CJ/(fid->P_success);
    fid->F_swap = F_swap_times_P_success/fid->P_success;
}

void calculate_cphase_fidelities_E_field(CphaseFidelities *fid,
                                         const SpinwaveAndFieldVector &sol,
                                         const std::vector<std::complex<double>> &noImpR,
                                         std::complex<double> tNoInteraction)
{
    const int numRealizations = sol.size();
    assert(numRealizations > 0 && "no realizations in the argument!");
    assert(noImpR.size() == numRealizations && "noImpR.size() != fid.size()");

    const int num_t_points = sol[0].E_without_scattering0.size();
    for (int i = 0; i < numRealizations; ++i) {
        const int num_t_points_i = sol[i].E_without_scattering0.size();
        if (num_t_points != num_t_points_i) {
            std::cout << "realization " << i << " has " << num_t_points_i
                      << ", while the first realization has " << num_t_points
                      << std::endl;
            assert(0 && "No all electric field arrays have the same lengths!");
        }
    }

    // Take the reference outgoing electric field mode
    // to be the average of the modes from each realization
    Eigen::VectorXcd E_without_scattering_avg = Eigen::VectorXcd::Zero(num_t_points);
    for (int i = 0; i < numRealizations; ++i) {
        E_without_scattering_avg += sol[i].E_without_scattering0;
    }
    // Dividing by numRealizations here is in principle not necessary
    // and is also not sufficient to properly normalize the averaged
    // state, since the individual solutions are not orthogonal.
    E_without_scattering_avg /= numRealizations;

    const double E_without_scattering_avg_squaredNorm = E_without_scattering_avg.squaredNorm();

    double P_success = 0;
    double F_swap_times_P_success = 0;
    double F_CJ = 0;
    double eta_EIT = 0;
    #pragma omp parallel for reduction(+:P_success,F_swap_times_P_success,F_CJ,eta_EIT)
    for (int i = 0; i < numRealizations; ++i) {
        const double E_without_scattering0_squaredNorm = sol[i].E_without_scattering0.squaredNorm();
        const double E_without_scattering1_squaredNorm = sol[i].E_without_scattering1.squaredNorm();
        const double E_with_scattering_squaredNorm = sol[i].E_with_scattering.squaredNorm();
        eta_EIT += E_without_scattering0_squaredNorm;
        P_success
                += 0.25*((std::norm(tNoInteraction)+std::norm(noImpR[i]))*E_without_scattering0_squaredNorm
                         +std::norm(tNoInteraction)*E_without_scattering1_squaredNorm
                         +E_with_scattering_squaredNorm);
        F_swap_times_P_success
                += ((tNoInteraction+noImpR[i])*sol[i].E_without_scattering0
                    +tNoInteraction*sol[i].E_without_scattering1
                    -sol[i].E_with_scattering).squaredNorm()/16;
        F_CJ += std::norm((tNoInteraction+noImpR[i])*E_without_scattering_avg.dot(sol[i].E_without_scattering0)
                          +tNoInteraction*E_without_scattering_avg.dot(sol[i].E_without_scattering1)
                          -E_without_scattering_avg.dot(sol[i].E_with_scattering))
                    /(16*E_without_scattering_avg_squaredNorm);
    }
    fid->P_success = P_success;
    fid->P_success /= numRealizations;
    F_swap_times_P_success /= numRealizations;
    fid->F_CJ = F_CJ;
    fid->F_CJ /= numRealizations;
    fid->F_CJ_conditional = fid->F_CJ/(fid->P_success);
    fid->F_swap = F_swap_times_P_success/fid->P_success;
    fid->single_photon_storage_retrieval_eff = eta_EIT;
    fid->single_photon_storage_retrieval_eff /= numRealizations;
}

namespace {
inline double squaredNorm_with_weights(const Eigen::VectorXcd &vec, const Eigen::ArrayXd &weights)
{
    const int vec_size = vec.size();
    assert(weights.size() == vec_size && "The vector and weights have different sizes!");
    double squaredNorm = 0;
    for (int i = 0; i < vec_size; ++i) {
        squaredNorm += std::norm(vec(i))*weights(i);
    }
    return squaredNorm;
}

inline std::complex<double> inner_product_with_weights(const Eigen::VectorXcd &v, const Eigen::VectorXcd &u, const Eigen::ArrayXd &weights)
{
    const int v_size = v.size();
    assert(u.size() == v_size && "The vectors have different sizes!");
    assert(weights.size() == v_size && "The vector and weights have different sizes!");
    std::complex<double> inner_product = 0;
    for (int i = 0; i < v_size; ++i) {
        inner_product += std::conj(v(i))*u(i)*weights(i);
    }
    return inner_product;
}
}

void calculate_cphase_fidelities_E_field_with_weights(CphaseFidelities *fid,
                                                      const SpinwaveAndFieldVector &sol,
                                                      const std::vector<std::complex<double>> &noImpR,
                                                      std::complex<double> tNoInteraction)
{
    Eigen::ArrayXd weights = sol.tWeights;
    const int numRealizations = sol.size();
    assert(numRealizations > 0 && "no realizations in the argument!");
    assert(noImpR.size() == numRealizations && "noImpR.size() != fid.size()");

    const int num_t_points = weights.size();
    for (int i = 0; i < numRealizations; ++i) {
        const int num_t_points_i = sol[i].E_without_scattering0.size();
        if (num_t_points != num_t_points_i) {
            std::cout << "realization " << i << " has " << num_t_points_i
                      << ", while the first realization has " << num_t_points
                      << std::endl;
            assert(0 && "No all electric field arrays have the same lengths!");
        }
    }

    // Take the reference outgoing electric field mode
    // to be the average of the modes from each realization
    Eigen::VectorXcd E_without_scattering_avg = Eigen::VectorXcd::Zero(num_t_points);
    for (int i = 0; i < numRealizations; ++i) {
        E_without_scattering_avg += sol[i].E_without_scattering0;
    }
    // Dividing by numRealizations here is in principle not necessary
    // and is also not sufficient to properly normalize the averaged
    // state, since the individual solutions are not orthogonal.
    E_without_scattering_avg /= numRealizations;

    const double E_without_scattering_avg_squaredNorm = squaredNorm_with_weights(E_without_scattering_avg, weights);

    double P_success = 0;
    double F_swap_times_P_success = 0;
    double F_CJ = 0;
    double eta_EIT = 0;
    #pragma omp parallel for reduction(+:P_success,F_swap_times_P_success,F_CJ,eta_EIT)
    for (int i = 0; i < numRealizations; ++i) {
        const double E_without_scattering0_squaredNorm = squaredNorm_with_weights(sol[i].E_without_scattering0, weights);
        const double E_without_scattering1_squaredNorm = squaredNorm_with_weights(sol[i].E_without_scattering1, weights);
        const double E_with_scattering_squaredNorm = squaredNorm_with_weights(sol[i].E_with_scattering, weights);
        eta_EIT += E_without_scattering0_squaredNorm;
        P_success
                += 0.25*((std::norm(tNoInteraction)+std::norm(noImpR[i]))*E_without_scattering0_squaredNorm
                         +std::norm(tNoInteraction)*E_without_scattering1_squaredNorm
                         +E_with_scattering_squaredNorm);
        F_swap_times_P_success
                += squaredNorm_with_weights(
                    (tNoInteraction+noImpR[i])*sol[i].E_without_scattering0
                     +tNoInteraction*sol[i].E_without_scattering1
                     -sol[i].E_with_scattering, weights)/16;
        F_CJ += std::norm((tNoInteraction+noImpR[i])*inner_product_with_weights(E_without_scattering_avg, sol[i].E_without_scattering0, weights)
                          +tNoInteraction*inner_product_with_weights(E_without_scattering_avg, sol[i].E_without_scattering1, weights)
                          -inner_product_with_weights(E_without_scattering_avg, sol[i].E_with_scattering, weights))
                    /(16*E_without_scattering_avg_squaredNorm);
    }
    fid->P_success = P_success;
    fid->P_success /= numRealizations;
    F_swap_times_P_success /= numRealizations;
    fid->F_CJ = F_CJ;
    fid->F_CJ /= numRealizations;
    fid->F_CJ_conditional = fid->F_CJ/(fid->P_success);
    fid->F_swap = F_swap_times_P_success/fid->P_success;
    fid->single_photon_storage_retrieval_eff = eta_EIT;
    fid->single_photon_storage_retrieval_eff /= numRealizations;
}

namespace {
double find_Delta_at_first_resonance_f(unsigned n, const double *x, double *grad, void *params)
{
   find_delta_at_first_resonance_params *p
            = (find_delta_at_first_resonance_params *) params;
    const double Delta = x[0];
    const double kd = p->kd_ensemble;
    RandTCoefficients rAndT = (*(p->f))(Delta, kd, kd,
                                        p->Deltac, p->g1d, p->OmegaScattering,
                                        p->NAtoms, p->NAtoms/2);
    if (std::isnan(rAndT.r.real())) {
        std::cout << "  rAndT.r = " << rAndT.r << std::endl;
        std::cout << "  kd_ensemble = " << p->kd_ensemble << std::endl;
        std::cout << "  NAtoms = " << p->NAtoms << std::endl;
        std::cout << "  Delta = " << Delta << std::endl;
        std::cout << "  Deltac = " << p->Deltac << std::endl;
    }
#ifdef CPHASE_GATE_DEBUG_PRINT
    //std::cout << "   delta = " << Delta-(p->Deltac) << ", |rAndT.r| = " << std::abs(rAndT.r) << std::endl;
#endif
    return std::abs(rAndT.r);
}
} // unnamed namespace

bool find_delta_at_first_resonance(find_delta_at_first_resonance_params *params)
{
    double delta_guess
            = approximate_resonance_solution1(params->g1d, params->NAtoms,
                                              params->Deltac,
                                              params->OmegaScattering);
    if (delta_guess > -params->Deltac/2) {
        delta_guess = -params->Deltac/2;
    }

    double delta_step = std::abs(delta_guess)/10;
    if (std::isnan(delta_guess)) {
        return false;
    }


    double delta_guess2 = delta_guess;
    // For a Dual-V ensemble the above guess is fine
    // since it behaves as an ordered Lambda-type
    // ensemble even if the atoms are disordered.
    // TODO: There is a factor of 2 in the mass though
    //       (It might not be such a big problem since
    //       the resonance frequency is found numerically
    //       below anyway).
    if (!(params->f->flags() & ENSEMBLE_SCATTERING_DUAL_V_ATOMS)) {
        // Now we refine our guess a bit. The reason
        // is that the first guess above is only good
        // for the case of the 3/2*pi (1/2*pi) setup
        // and becomes increasingly different from the
        // actual resonance Delta value. It becomes a
        // problem if the guess is closer to some other
        // resonance of the ensemble "cavity" instead of
        // first one. (For example the "zeroth" resonance,
        // i.e Delta=Deltac, which only appears in the
        // case of the random placement of the atoms.)
        // Thus we do refinement using the knowledge about
        // the resonances of our ensemble based "cavity".
        // They occur when
        //    sin(NAtoms*qd)=0,
        // where qd is the Bloch vector multiplied by the
        // spacing between the atoms d. We also ignore the
        // fact that qd can be complex and only look at the
        // real part, since it is the real part that gives
        // the oscillatory behaviour and the imaginary part
        // is responsible for the loss mechanisms.
        // The solutions to the equation above are given by
        //    NAtoms*qd = m*pi,
        // where m is an integer. Since qd itself is given by
        //    qd = arccos(tr(Mcell)/2)/periodLength,
        // it we have 0 <= qd <= pi/periodLength. Thus it
        // must hold that
        //    0 <= m <= NAtoms/periodLength = NUnitCells.
        // What we called the "zeroth" resonance above
        // occurs at m=0 or m=NUnitCells, depending on the
        // setup (i.e. how atoms are positioned). And the
        // "first" resonance is either m=1 or m=NUnitCells-1.
        // For the 3/2*pi setup the "first" resonance is at
        // m=NUnitCells-1. This is how the code in
        // "approximate_resonance_solution1" calculates it.
        //
        // The bottomline of the above discussion is that we just
        // need to solve the equation
        //    NAtoms*qd = m*pi = (NUnitCells-1)*pi
        // <=>
        //    NAtoms*(qd-pi/periodLength) = (NUnitCells-1)*pi-NAtoms*pi/periodLength
        // <=>
        //    NAtoms*(qd-pi/periodLength) = -pi
        // <=>
        //    qd-pi/periodLength = -pi/NAtoms
        //
        // (In the code we also take the absolute value on
        // both sides.)

        int periodLength = find_period_length_from_kd(params->kd_ensemble);
        const bool disorderedEnsemble = (periodLength < 0)
                || (params->f->flags() & ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS);
        if (disorderedEnsemble) {
            // For Lambda-type atoms, the ensemble which is not
            // periodic behaves for big range of parameters as
            // a periodic ensemble with very high periodicity.
            // Hence choose some big number here.
            // TODO: In principle, for some parameters this value
            //       may need to be even higher to be a good
            //       approximation for the ensemble without periodicity
            periodLength = 32;
        }
        OptimalDeltaFromDispersionRelation funcDispersionRelation(
                    params->Deltac, params->OmegaScattering, params->g1d,
                    periodLength, params->NAtoms);
        delta_guess2 = find_root_secant<double>(funcDispersionRelation,
                                                delta_guess,
                                                0.9*delta_guess,
                                                1e-8);
    }
    if (std::isinf(delta_guess2) || std::isnan(delta_guess2)) {
        std::cout << " delta_guess2 = " << delta_guess2
                  << ", delta_guess1 = " << delta_guess
                  << ", Deltac = " << params->Deltac << std::endl;
    }
    delta_guess = delta_guess2;
    delta_step = std::abs(delta_guess)/10;
    // We do not want to start too close to the
    // one-photon resonance (delta_guess = -params->Deltac)
    // since the transfer matrix reflection and transmission
    // calculation is often not numerically stable in that
    // region
    if (delta_guess > -params->Deltac/2) {
        delta_guess = -params->Deltac/2;
        delta_step = std::abs(delta_guess)/10;
    }

    const int numParams = 1;
    nlopt::opt opt(nlopt::LN_SBPLX, numParams);
    std::vector<double> x(numParams);
    x[0] = delta_guess;
    std::vector<double> step(numParams);
    step[0] = delta_step;
    opt.set_initial_step(step);

    std::vector<double> lb(numParams);
    lb[0] = 0;
    opt.set_lower_bounds(lb);

    std::vector<double> ub(numParams);

    // delta = -Delta_c <=> Delta = 0
    ub[0] = -params->Deltac;

    opt.set_upper_bounds(ub);

    opt.set_min_objective(find_Delta_at_first_resonance_f, params);
    opt.set_xtol_abs(1e-14);

    double minf;
    nlopt::result result = opt.optimize(x, minf);
    params->delta = x[0];
#ifdef CPHASE_GATE_DEBUG_PRINT
    //std::cout << "   delta = " << params->delta << std::endl;
#endif

    return true;
}

void calculate_cphase_fidelities_for_tNoInteraction(
        const SpinwaveAndFieldVector &sol,
        CphaseFidelities *fid,
        const std::vector<std::complex<double>> &noImpR,
        std::complex<double> tNoInteraction,
        int cphaseGateFlags,
        bool onlyAnalyticalCalculation)
{
    if (cphaseGateFlags & CPHASE_GATE_PROJECT_SPINWAVES) {
        calculate_cphase_fidelities_spinwave(fid, sol, noImpR,
                                             tNoInteraction);
    } else {
        if (onlyAnalyticalCalculation) {
            calculate_cphase_fidelities_E_field_with_weights(fid, sol, noImpR,
                                                             tNoInteraction);
        } else {
            calculate_cphase_fidelities_E_field(fid, sol, noImpR,
                                                tNoInteraction);
        }
    }
}

namespace {
struct find_optimal_Deltac_sigma_params
{
    EnsembleScattering *ensemble_scattering_regular;
    double kd_ensemble_regular;
    EnsembleScattering *ensemble_scattering;
    CphaseFidelities *fid_tNoInteraction_one;
    SpinwaveAndFieldVector sol;
    std::vector<std::complex<double>> noImpR;
    CphaseGateFidelityParameters *cphaseParameters;
    int cphaseGateFlags;
    bool onlyAnalyticalCalculation;
    bool stopWhenCondFidTolReached;
    std::vector<QuadratureK_r> quadK_r;
};

void calculate_cphase_fidelities_numerical_wrapper(
        find_optimal_Deltac_sigma_params *p)
{
    const int numRealizations = p->cphaseParameters->randomSeeds.size();
    calculate_store_eit_multiply_reflection_coefficient_retrieve(
                p->sol, p->cphaseParameters,
                p->cphaseGateFlags,
                p->onlyAnalyticalCalculation,
                p->ensemble_scattering,
                p->quadK_r);
    for (int i = 0; i < numRealizations; ++i) {
        const unsigned long int randomSeed
                = p->cphaseParameters->randomSeeds[i];
        p->ensemble_scattering->setRandomSeed(randomSeed);
        const std::complex<double> noImpR = no_impurity_reflection_coefficient(
                    p->cphaseParameters->delta,
                    p->cphaseParameters->g1d,
                    p->cphaseParameters->NAtoms,
                    p->cphaseParameters->Deltac,
                    p->cphaseParameters->OmegaScattering,
                    p->cphaseParameters->kd_ensemble,
                    p->cphaseParameters->kL1,
                    p->cphaseParameters->kL2,
                    p->ensemble_scattering,
                    p->cphaseGateFlags);
        p->noImpR[i] = noImpR;
    }
    calculate_cphase_fidelities_for_tNoInteraction(p->sol,
                                                   p->fid_tNoInteraction_one,
                                                   p->noImpR, 1,
                                                   p->cphaseGateFlags,
                                                   p->onlyAnalyticalCalculation);
}

void retrieve_stored_spinwave_analytical_for_realization(
        SpinwaveAndFieldVector &sol,
        const HamiltonianParams &eitParams,
        const Eigen::MatrixXcd &K_r,
        const ThreadedEigenMatrix &K_r_threaded,
        const int n,
        bool dualVAtoms,
        bool symmetricStorage,
        bool parallelizeOverRealizations)
{
    const int NAtoms = eitParams.NAtoms;
#ifdef CPHASE_GATE_DEBUG_PRINT
    //std::cout << " n = " << n << ", dtRetrieval = " << dtRetrieval[n] << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT

    int N_t = sol.tVals.size();

    sol[n].E_without_scattering0 = Eigen::VectorXcd::Zero(N_t);
    sol[n].E_with_scattering = Eigen::VectorXcd::Zero(N_t);
    sol[n].E_R_without_scattering0 = Eigen::VectorXcd::Zero(N_t);
    sol[n].E_L_without_scattering0 = Eigen::VectorXcd::Zero(N_t);
    sol[n].E_R_with_scattering = Eigen::VectorXcd::Zero(N_t);
    sol[n].E_L_with_scattering = Eigen::VectorXcd::Zero(N_t);

    if (dualVAtoms && !symmetricStorage) {
        const double dz = 1.0/NAtoms;
        if (parallelizeOverRealizations) {
            sol[n].E_without_scattering0 = K_r*sol[n].psi_after_storage_analytical_plus_ikz*std::sqrt(dz);
            sol[n].E_with_scattering = K_r*sol[n].psi_after_scattering_analytical_plus_ikz*std::sqrt(dz);
        } else { // parallelize over N_t
            sol[n].E_without_scattering0 = K_r_threaded*sol[n].psi_after_storage_analytical_plus_ikz*std::sqrt(dz);
            sol[n].E_with_scattering = K_r_threaded*sol[n].psi_after_scattering_analytical_plus_ikz*std::sqrt(dz);
        }
    } else {
        const double dz = 1.0/NAtoms;
        Eigen::VectorXcd psi_after_storage_analytical_minus_ikz_reverse(NAtoms);
        Eigen::VectorXcd psi_after_scattering_analytical_minus_ikz_reverse(NAtoms);
        for (int k = 0; k < NAtoms; ++k) {
            if (k != 0) {
                psi_after_storage_analytical_minus_ikz_reverse(k)
                        = sol[n].psi_after_storage_analytical_minus_ikz(NAtoms-k);
                psi_after_scattering_analytical_minus_ikz_reverse(k)
                        = sol[n].psi_after_scattering_analytical_minus_ikz(NAtoms-k);
            } else {
                psi_after_storage_analytical_minus_ikz_reverse(k) = 0;
                psi_after_scattering_analytical_minus_ikz_reverse(k) = 0;
            }
        }
        if (parallelizeOverRealizations) {
            sol[n].E_R_without_scattering0 = K_r*sol[n].psi_after_storage_analytical_plus_ikz*M_SQRT1_2*std::sqrt(dz);
            sol[n].E_R_with_scattering = K_r*sol[n].psi_after_scattering_analytical_plus_ikz*M_SQRT1_2*std::sqrt(dz);
            sol[n].E_L_without_scattering0 = K_r*psi_after_storage_analytical_minus_ikz_reverse*M_SQRT1_2*std::sqrt(dz);
            sol[n].E_L_with_scattering = K_r*psi_after_scattering_analytical_minus_ikz_reverse*M_SQRT1_2*std::sqrt(dz);
        } else { // parallelize over NAtoms
            sol[n].E_R_without_scattering0 = K_r_threaded*sol[n].psi_after_storage_analytical_plus_ikz*M_SQRT1_2*std::sqrt(dz);
            sol[n].E_R_with_scattering = K_r_threaded*sol[n].psi_after_scattering_analytical_plus_ikz*M_SQRT1_2*std::sqrt(dz);
            sol[n].E_L_without_scattering0 = K_r_threaded*psi_after_storage_analytical_minus_ikz_reverse*M_SQRT1_2*std::sqrt(dz);
            sol[n].E_L_with_scattering = K_r_threaded*psi_after_scattering_analytical_minus_ikz_reverse*M_SQRT1_2*std::sqrt(dz);
        }
    }
    sol[n].psi_after_retrieval_with_scattering = Eigen::VectorXcd::Zero(2*NAtoms);
    if (dualVAtoms && !symmetricStorage) {
        sol[n].E_R_without_scattering0 = sol[n].E_without_scattering0;
        sol[n].E_L_without_scattering0 = Eigen::VectorXcd::Zero(N_t);
        sol[n].E_R_with_scattering = sol[n].E_with_scattering;
        sol[n].E_L_with_scattering = Eigen::VectorXcd::Zero(N_t);
    } else {
        sol[n].E_with_scattering
                = 1.0/std::sqrt(2)*(sol[n].E_R_with_scattering + sol[n].E_L_with_scattering);
        sol[n].E_without_scattering0
                = 1.0/std::sqrt(2)*(sol[n].E_R_without_scattering0 + sol[n].E_L_without_scattering0);
    }
    sol[n].E_without_scattering1 = sol[n].E_without_scattering0;
}

void retrieve_stored_spinwave_numerical_for_realization(
        SpinwaveAndFieldVector &sol,
        const HamiltonianParams &eitParams,
        const std::vector<double> &atomPositions0,
        const std::vector<double> &atomPositions1,
        double dtRetrieval,
        double t_retrieval,
        EvolutionMethod method,
        int flags,
        const int n,
        bool dualVAtoms,
        bool symmetricStorage,
        bool randomPlacement)
{
    HamiltonianParams eitParams0 = eitParams;
    eitParams0.atom_positions = atomPositions0;
    LambdaHamiltonian1Excitation H_EIT;
    H_EIT.setParams(eitParams0);

    const double dtSqrt = std::sqrt(dtRetrieval);
    sol[n].psi_after_retrieval_without_scattering
            = H_EIT.evolve(sol[n].psi_after_storage0, t_retrieval, method,
                           flags, dtRetrieval);
    sol[n].E_R_without_scattering0 = H_EIT.E_R();
    sol[n].E_L_without_scattering0 = H_EIT.E_L();


    HamiltonianParams eitParams1 = eitParams;
    eitParams1.atom_positions = atomPositions1;
    H_EIT.setParams(eitParams1);
    Eigen::VectorXcd E_R_without_scattering1;
    Eigen::VectorXcd E_L_without_scattering1;
    if (randomPlacement) {
        H_EIT.evolve(sol[n].psi_after_storage1, t_retrieval, method, flags,
                     dtRetrieval);
        E_R_without_scattering1 = H_EIT.E_R();
        E_L_without_scattering1 = H_EIT.E_L();
    } else {
        E_R_without_scattering1 = sol[n].E_R_without_scattering0;
        E_L_without_scattering1 = sol[n].E_L_without_scattering0;
    }
    sol[n].psi_after_retrieval_with_scattering
                = H_EIT.evolve(sol[n].psi_after_scattering, t_retrieval, method,
                               flags, dtRetrieval);
    sol[n].E_R_with_scattering = H_EIT.E_R();
    sol[n].E_L_with_scattering = H_EIT.E_L();

    sol.tVals = H_EIT.tVals();

    sol[n].E_R_with_scattering *= dtSqrt;
    sol[n].E_L_with_scattering *= dtSqrt;
    sol[n].E_R_without_scattering0 *= dtSqrt;
    sol[n].E_L_without_scattering0 *= dtSqrt;
    E_R_without_scattering1 *= dtSqrt;
    E_L_without_scattering1 *= dtSqrt;

    if (dualVAtoms && !symmetricStorage) {
        // For the dual-V atoms, we have stored a right-going
        // photon in the ensemble (incident from the left).
        // Upon retrieving, the stored excitation just
        // continues propagating through the ensemble and
        // exits on the right side.
        sol[n].E_with_scattering = sol[n].E_R_with_scattering;
        sol[n].E_without_scattering0 = sol[n].E_R_without_scattering0;
        sol[n].E_without_scattering1 = E_R_without_scattering1;
    } else {
        // For the Lambda-type atoms (or dual-V atoms when
        // explicitly requested to do symmetric storage) we have
        // stored two halves of the photon that were incident
        // from both sides of the ensemble. Hence we need to
        // effectively put them both through a beam splitter to
        // recover the whole excitation in one mode.
        sol[n].E_with_scattering
                = 1.0/std::sqrt(2)*(sol[n].E_R_with_scattering + sol[n].E_L_with_scattering);
        sol[n].E_without_scattering0
                = 1.0/std::sqrt(2)*(sol[n].E_R_without_scattering0 + sol[n].E_L_without_scattering0);
        sol[n].E_without_scattering1
                = 1.0/std::sqrt(2)*(E_R_without_scattering1 + E_L_without_scattering1);
    }
}

void retrieve_stored_spinwave(
        SpinwaveAndFieldVector &sol,
        CphaseGateFidelityParameters *cphaseParameters,
        bool onlyAnalyticalCalculation,
        bool dualVAtoms,
        bool symmetricStorage,
        bool randomPlacement,
        std::vector<std::vector<double>> atomPositionsAllRealizations0,
        std::vector<std::vector<double>> atomPositionsAllRealizations1,
        std::complex<double> noImpR,
        std::vector<QuadratureK_r> &quadK_r,
        HamiltonianParams &eitParams,
        double dtInitial,
        double t_retrieval,
        EvolutionMethod method,
        double normThreshold,
        double fidThreshold,
        int flags,
        int numRealizations)
{
    const int NAtoms = cphaseParameters->NAtoms;

    double dtRetrieval = dtInitial;
    double F_CJ = 0;
    double E_without_scattering_norm = 0;
    const int num_dt_levels = 20;
    const int numThreads = omp_get_max_threads();
    const bool parallelizeOverRealizations = (numRealizations > numThreads);
    for (int i = 0; i < num_dt_levels; ++i) {
        if (onlyAnalyticalCalculation) {
            if (quadK_r.size() > 0 && quadK_r[0].NAtoms != NAtoms) {
                // The cache has been computed for an ensemble
                // with a different number of atoms -- clear it.
                std::cout << "Clearing quadrature cache" << std::endl;
                quadK_r.clear();
            }
            if (i < quadK_r.size()) {
#ifdef CPHASE_GATE_DEBUG_PRINT
                std::cout << " Using cache" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
                sol.tVals = quadK_r[i].tVals;
                sol.tWeights = quadK_r[i].tWeights;
            } else {
#ifdef CPHASE_GATE_DEBUG_PRINT
                std::cout << " Computing tanh-sinh nodes and weights" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
                // Ensure that the maximal degree is at least 3
                const int max_degree = i + 3;
                std::vector<Node> all_nodes = calc_nodes_tanh_sinh_final(0, t_retrieval, max_degree);
                int N_t = all_nodes.size();
                sol.tVals = Eigen::ArrayXd::Zero(N_t);
                sol.tWeights = Eigen::ArrayXd::Zero(N_t);
                for (int i = 0; i < N_t; ++i) {
                    sol.tVals(i) = all_nodes[i].node;
                    //std::cout << " tVals(" << i << ") = " << sol[n].tVals(i) << std::endl;
                    sol.tWeights(i) = all_nodes[i].weight;
                }

#ifdef CPHASE_GATE_DEBUG_PRINT
                std::cout << " Computing K_r" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
                const double g1d = eitParams.g1d;
                const double Omega = eitParams.Omega;
                const std::complex<double> gprimeDelta = (1-g1d)/2;
                const double g1dN = g1d/2*NAtoms;
                const double Omega2 = Omega*Omega;
                QuadratureK_r quadK_r_element;
                auto K_r_element_func = [=] (int j, int k) -> std::complex<double> {
                        const double tj = sol.tVals(j);
                        const double z = static_cast<double>(k)/NAtoms;
                        return std::sqrt(gprimeDelta)/(2.0*std::sqrt(M_PI)*std::pow(Omega2*tj*g1dN*(1-z), 0.25))
                          *std::exp(-std::pow(std::sqrt(Omega2*tj)-std::sqrt(g1dN*(1-z)),2)/gprimeDelta)
                          *std::sqrt(g1d/(2))*std::sqrt(NAtoms)*Omega/gprimeDelta;

                    };
                if (parallelizeOverRealizations) {
                    quadK_r_element.K_r = Eigen::MatrixXcd::Zero(N_t, NAtoms);
                    #pragma omp parallel for
                    for (int j = 0; j < N_t; ++j) {
                        for (int k = 0; k < NAtoms; ++k) {
                            quadK_r_element.K_r(j,k) = K_r_element_func(j,k);
                        }
                    }
                    //std::cout << "K_r size = " << quadK_r_element.K_r.rows() << " x " << quadK_r_element.K_r.cols() << std::endl;
                } else { // parallelize over N_t
                    quadK_r_element.K_r_threaded = ThreadedEigenMatrix(K_r_element_func, N_t, NAtoms);
                }
                quadK_r_element.tVals = sol.tVals;
                quadK_r_element.tWeights = sol.tWeights;
                quadK_r_element.NAtoms = NAtoms;
                quadK_r.push_back(quadK_r_element);
            }
        }
        if (parallelizeOverRealizations) {
#ifdef CPHASE_GATE_DEBUG_PRINT
            std::cout << " Computing retrieved electric field (parallelize over realizations)" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
            // If there are many realizations, it is better to
            // parallelize at this level. The functions that
            // are called in this loop can ultimately also do
            // parallelization of their own, but since it is more
            // fine grained, it also has more overhead.
            #pragma omp parallel for
            for (int n = 0; n < numRealizations; ++n) {
                if (onlyAnalyticalCalculation) {
                    retrieve_stored_spinwave_analytical_for_realization(
                                sol, eitParams,
                                quadK_r[i].K_r,
                                quadK_r[i].K_r_threaded,
                                n, dualVAtoms, symmetricStorage,
                                parallelizeOverRealizations);
                } else {
                    retrieve_stored_spinwave_numerical_for_realization(
                                sol, eitParams,
                                atomPositionsAllRealizations0[n],
                                atomPositionsAllRealizations1[n],
                                dtRetrieval, t_retrieval,
                                method, flags, n, dualVAtoms, symmetricStorage,
                                randomPlacement);
                }
            }
        } else { // parallelize over N_t
#ifdef CPHASE_GATE_DEBUG_PRINT
            std::cout << " Computing retrieved electric field (parallelize over NAtoms)" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
            // This is exactly the same loop as above, but without
            // the OpenMP #pragma to make it parallel.
            for (int n = 0; n < numRealizations; ++n) {
                if (onlyAnalyticalCalculation) {
                    retrieve_stored_spinwave_analytical_for_realization(
                                sol, eitParams,
                                quadK_r[i].K_r,
                                quadK_r[i].K_r_threaded,
                                n, dualVAtoms, symmetricStorage,
                                parallelizeOverRealizations);
                } else {
                    retrieve_stored_spinwave_numerical_for_realization(
                                sol, eitParams,
                                atomPositionsAllRealizations0[n],
                                atomPositionsAllRealizations1[n],
                                dtRetrieval, t_retrieval,
                                method, flags, n, dualVAtoms, symmetricStorage,
                                randomPlacement);
                }
            }
        }
        CphaseFidelities fid;

        std::vector<std::complex<double>> noImpRArray(numRealizations, noImpR);
        double E_without_scattering_norm_new;
        if (onlyAnalyticalCalculation) {
            calculate_cphase_fidelities_E_field_with_weights(&fid, sol, noImpRArray, 1);
        } else {
            calculate_cphase_fidelities_E_field(&fid, sol, noImpRArray, 1);
        }
        E_without_scattering_norm_new = std::sqrt(fid.single_photon_storage_retrieval_eff);
        const double diff = std::abs(E_without_scattering_norm_new - E_without_scattering_norm);
        const double diff_F = std::abs(F_CJ - fid.F_CJ);
#ifdef CPHASE_GATE_DEBUG_PRINT
        std::cout << " E_without_scattering_norm = " << E_without_scattering_norm_new
                  << " (diff = " << diff << ")"
                  << "; F_CJ = " << fid.F_CJ
                  << " (diff_F = " << diff_F << ")"
                  << "; F_swap = " << fid.F_swap
                  << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
        if (diff < normThreshold && diff_F < fidThreshold) {
            if (E_without_scattering_norm > 1) {
                std::cout << "Retrieved field (without scattering) has norm = "
                          << E_without_scattering_norm << std::endl;
                assert(0 && "Retrieved field (without scattering) has norm bigger than unity.");
            }
            break;
        }
        if (i == num_dt_levels-1) {
            std::cout << " dt was halved " << num_dt_levels
                      << " times, and the errors are still above tolerance." << '\n';
            std::cout << " E_without_scattering_norm = " << E_without_scattering_norm_new
                      << " (diff = " << diff << ")"
                      << "; F_CJ = " << fid.F_CJ
                      << " (diff_F = " << diff_F << ")"
                      << "; F_swap = " << fid.F_swap
                      << std::endl;
        }
        F_CJ = fid.F_CJ;
        E_without_scattering_norm = E_without_scattering_norm_new;

        dtRetrieval /= 2;
    }
}

std::vector<std::complex<double>> calculate_imp_reflection_coefficient_array(
        CphaseGateFidelityParameters *cphaseParameters,
        int cphaseGateFlags,
        EnsembleScattering *ensemble_scattering)
{
    const int NAtoms = cphaseParameters->NAtoms;
    const double g1d = cphaseParameters->g1d;
    const double kd_ensemble = cphaseParameters->kd_ensemble;
    std::vector<std::complex<double>> R(NAtoms);
    // It is important to fill the internal class array for
    // ensemble_scattering in a single thread. Otherwise if
    // all the OpenMP threads in the loop below start calling
    // the same object, it will create a race condition.
    ensemble_scattering->fillAtomArraysNewValues(
                NAtoms, kd_ensemble, kd_ensemble);
    ensemble_scattering->fillTransferMatrixArrayNewValues(
                cphaseParameters->delta, cphaseParameters->Deltac,
                cphaseParameters->g1d, cphaseParameters->OmegaScattering);
    // Multiply with the reflection coefficient of
    // the second photon
    #pragma omp parallel for
    for (int i = 0; i < NAtoms; ++i) {
        R[i] = impurity_reflection_coefficient_discrete(
                    i, cphaseParameters->delta, g1d,
                    NAtoms, cphaseParameters->Deltac,
                    cphaseParameters->OmegaScattering,
                    cphaseParameters->kd_ensemble,
                    cphaseParameters->kL1,
                    cphaseParameters->kL2,
                    ensemble_scattering,
                    cphaseGateFlags);
    }
    return R;
}
} // unnamed namespace

void set_stored_spin_wave_set_tolerances(double &dtInitial,
                                         double &normThreshold,
                                         double &fidThreshold,
                                         bool onlyAnalyticalCalculation)
{
    if (onlyAnalyticalCalculation) {
        dtInitial = 0.5;
        // For the analytical calculation we can afford
        // tighter tolerances.
        normThreshold = 1e-10;
        fidThreshold = 1e-10;
    } else {
        // For the numerical calculation, Runge-Kutta evolution
        // is used. And there the step size should be small
        // enough to get an answer, which is not a NaN.
        // On the other hand, even the same step size can be
        // used here without any problem. The only effect is that
        // the first iterations will yield a NaN and the code will
        // automatically halve the time step  until it is no longer
        // the case.
        dtInitial = 0.25;

        normThreshold = 1e-6;
        fidThreshold = 1e-6;
    }
}

void set_optmization_tolerance(nlopt::opt &optimization, bool onlyAnalyticalCalculation)
{
    // This sort of a heuristic distinction. The motivation
    // is that for analytical storage/retrieval, the tolerance
    // for the individual points is 1e-10, so that choosing
    // 1e-9 as the final tolerance for the optimization
    // makes sense.
    // On the other hand, for the numerical calculations,
    // the tolerance for the individual points is set to 1e-6,
    // since otherwise it will take too long to calculate
    // the individual points (requiring a lot of time steps
    // for the Runge-Kutta evolution). Hence here we rather
    // optimize for the tolerance in the arguments, since
    // that what we have been doing for a long time and it
    // worked well.
    if (onlyAnalyticalCalculation) {
        optimization.set_ftol_abs(1e-9);
    } else {
        optimization.set_xtol_rel(1e-8);
    }
}


Eigen::VectorXcd calculate_stored_spinwave_eit_field_elimination(
        LambdaHamiltonian1Excitation &H_EIT,
        EvolutionMethod method,
        double t_storage,
        double dtInitial,
        double normThreshold,
        int flags)
{
    const int NAtoms = H_EIT.params()->NAtoms;
    Eigen::VectorXcd psi_initial = Eigen::VectorXcd::Zero(2*NAtoms);
    Eigen::VectorXcd psi_final;
    double dtStorage = dtInitial;
    double psi_stored_norm = 0;
    for (int i = 0; i < 10; ++i) {
        psi_final = H_EIT.evolve(psi_initial, t_storage, method, flags, dtStorage);
        const double psi_stored_norm_new = psi_final.norm();
        const double diff = std::abs(psi_stored_norm_new - psi_stored_norm);
        if (diff < normThreshold) {
            break;
        }
        psi_stored_norm = psi_stored_norm_new;
        dtStorage /= 2;
    }
    // Zero out the excited states (|b>)
    for (int i = 0; i < NAtoms; ++i) {
        psi_final(i) = 0;
    }
    return psi_final;
}

std::vector<std::complex<double>>
calculate_stored_spinwave_eit_dispersion_relation_plus(
        HamiltonianParams eitParams,
        bool forceRegularPositions)
{
    std::vector<std::complex<double>> S_plus_array(eitParams.NAtoms);

    const std::complex<double> I(0,1);
    const double v_g_eit = 2*eitParams.gridSpacing*pow(eitParams.Omega, 2)/eitParams.g1d;
    const double alphaI = 4.0*(1-eitParams.g1d)*pow(eitParams.gridSpacing*eitParams.Omega/eitParams.g1d, 2);
    const double timeToPassEnsemble = 1.0/v_g_eit;
    const double initialEITPropagation = 0.5;
    const double initialMean = 0.0;
    const double t_EIT = initialEITPropagation*timeToPassEnsemble;
    const double sigmaInitial = eitParams.inputE_width*v_g_eit;
    // schedule(static, 1) seems to distribute the workload
    // more uniformly among threads
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < eitParams.NAtoms; ++i) {
        double z;
        if (forceRegularPositions) {
            z = static_cast<double>(i)/eitParams.NAtoms;
        } else {
            z = eitParams.atom_positions[i];
        }
        S_plus_array[i]
                = gaussian_final(z-initialMean, sigmaInitial,
                                 t_EIT, v_g_eit, -I*alphaI);
    }
    return S_plus_array;
}

std::vector<std::complex<double>>
calculate_stored_spinwave_eit_dispersion_relation_minus(
        HamiltonianParams eitParams,
        bool forceRegularPositions)
{
    std::vector<std::complex<double>> S_minus_array(eitParams.NAtoms);

    const std::complex<double> I(0,1);
    const double v_g_eit = 2*eitParams.gridSpacing*pow(eitParams.Omega, 2)/eitParams.g1d;
    const double alphaI = 4.0*(1-eitParams.g1d)*pow(eitParams.gridSpacing*eitParams.Omega/eitParams.g1d, 2);
    const double timeToPassEnsemble = 1.0/v_g_eit;
    const double initialEITPropagation = 0.5;
    const double initialMean = 0.0;
    const double t_EIT = initialEITPropagation*timeToPassEnsemble;
    const double sigmaInitial = eitParams.inputE_width*v_g_eit;
    // schedule(static, 1) seems to distribute the workload
    // more uniformly among threads
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < eitParams.NAtoms; ++i) {
        double z;
        if (forceRegularPositions) {
            z = static_cast<double>(i)/eitParams.NAtoms;
        } else {
            z = eitParams.atom_positions[i];
        }
        S_minus_array[i]
                = gaussian_final(z-(1-initialMean), sigmaInitial,
                                 t_EIT, -v_g_eit, -I*alphaI);
    }
    return S_minus_array;
}

std::vector<std::complex<double>>
calculate_stored_spinwave_eit_kernel_plus(
        HamiltonianParams eitParams,
        double t_storage,
        double storageTolAbs,
        double storageTolRel,
        bool forceRegularPositions)
{
    std::vector<std::complex<double>> S_plus_array(eitParams.NAtoms);
    // schedule(static, 1) seems to distribute the workload
    // more uniformly among threads
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < eitParams.NAtoms; ++i) {
        double z;
        if (forceRegularPositions) {
            z = static_cast<double>(i)/eitParams.NAtoms;
        } else {
            z = eitParams.atom_positions[i];
        }
        double absErr;
        double relErr;
        S_plus_array[i]
                = adiabatic_eit_store_spinwave_from_electric_field_asymptotic(
                    [=] (double t) -> std::complex<double>
        {
            return gaussian_input_electric_field_mode(t, eitParams.inputE_mean,
                                                      eitParams.inputE_width);
        }, z, t_storage, eitParams.g1d, eitParams.NAtoms, 0, 0, eitParams.Omega,
        storageTolAbs, storageTolRel, &absErr, &relErr);
        if (absErr > storageTolAbs) {
            std::cout << "absolute error = " << absErr
                      << " is bigger than absolute tolerance = "
                      << storageTolAbs
                      << ", z = " << z << std::endl;
        }
    }
    return S_plus_array;
}

std::vector<std::complex<double>>
calculate_stored_spinwave_eit_kernel_minus(
        HamiltonianParams eitParams,
        double t_storage,
        double storageTolAbs,
        double storageTolRel,
        bool forceRegularPositions)
{
    std::vector<std::complex<double>> S_minus_array(eitParams.NAtoms);
    // schedule(static, 1) seems to distribute the workload
    // more uniformly among threads
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < eitParams.NAtoms; ++i) {
        double z;
        if (forceRegularPositions) {
            z = static_cast<double>(i)/eitParams.NAtoms;
        } else {
            z = eitParams.atom_positions[i];
        }
        double absErr;
        double relErr;
        S_minus_array[i]
                = adiabatic_eit_store_spinwave_from_electric_field_asymptotic(
                    [=] (double t) -> std::complex<double>
        {
            return gaussian_input_electric_field_mode(t, eitParams.inputE_mean,
                                                      eitParams.inputE_width);
        }, 1-z, t_storage, eitParams.g1d, eitParams.NAtoms, 0, 0, eitParams.Omega,
        storageTolAbs, storageTolRel, &absErr, &relErr);
        if (absErr > storageTolAbs) {
            std::cout << "absolute error = " << absErr
                      << " is bigger than absolute tolerance = "
                      << storageTolAbs
                      << ", z = " << z << std::endl;
        }
    }
    return S_minus_array;
}

void calculate_store_eit_multiply_reflection_coefficient_retrieve(
        SpinwaveAndFieldVector &sol,
        CphaseGateFidelityParameters *cphaseParameters,
        int cphaseGateFlags, bool onlyAnalyticalCalculation,
        EnsembleScattering *ensemble_scattering,
        std::vector<QuadratureK_r> &quadK_r)
{
    // We use the ensemble_scattering to determine, whether
    // the atoms are randomly placed here instead of cphaseGateFlags
    // because they can be set to different values. See
    // calculate_cphase_fidelities_numerical_for_optimal_Deltac_sigma()
    // that ultimately ends up calling this function for the details.
    // In short: for the randomly placed ensembles we can do the
    // optimization with the regularly placed ensembles and then use
    // the optimal values to compute the fidelity with the actual randmly
    // placed ensemble. To make this temporary switch we create a separate
    // ensemble_scattering object with regularly placed atoms while we
    // do not change the value of cphaseGateFlags that are then passed
    // down to this function.
    const bool randomPlacement
            = ensemble_scattering->flags()
              & ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS;

    int numRealizations = cphaseParameters->randomSeeds.size();
    if (!randomPlacement) {
        // If the atoms are not randomly placed then the
        // random seeds array can be allowed to be empty
        // as there is always one and only one realization
        // to compute in this case.
        numRealizations = 1;
    }
    assert(numRealizations > 0 && "No ensemble realizations to compute!");
    if (sol.size() != numRealizations) {
        sol.resize(numRealizations);
    }
    const std::complex<double> I(0,1);
    const int NAtoms = cphaseParameters->NAtoms;
    const double g1d = cphaseParameters->g1d;
    const double kd_ensemble = cphaseParameters->kd_ensemble;
    // Target sigma of the spinwave
    const double sigma = cphaseParameters->sigma;

    HamiltonianParams eitParams;
    eitParams.NAtoms = NAtoms;
    const double L = 1; // Actually L / L
    eitParams.gridSpacing = grid_spacing(eitParams.NAtoms, L); // Actually gridSpacing / L
    eitParams.kd_ensemble = kd_ensemble;
    eitParams.g1d = g1d; // Actually Gamma_1D / (Gamma_1D + Gamma')

    eitParams.evolutionMethod = EvolutionMethod::RK4;


    // We want the EIT storage to be on
    // resonance. The values of delta and Deltac
    // passed to this function are meant for the
    // reflection coefficient of the second photon.
    eitParams.Deltac = 0;
    eitParams.delta = 0;

    eitParams.Omega = cphaseParameters->OmegaStorageRetrieval;
    const double v_g_eit = 2*eitParams.gridSpacing*pow(eitParams.Omega, 2)/eitParams.g1d;
    const double alphaI = 4.0*(1-eitParams.g1d)*pow(eitParams.gridSpacing*eitParams.Omega/eitParams.g1d, 2);
    const double timeToPassEnsemble = 1.0/v_g_eit;
    cphaseParameters->t_to_pass_ensemble = timeToPassEnsemble;

    const double initialEITPropagation = 0.5;
    const double initialMean = 0.0;
    const double t_EIT = initialEITPropagation*timeToPassEnsemble;
    const double sigmaInitial = sigma/std::sqrt(1+alphaI*t_EIT/(2*sigma*sigma));
    const bool dualVAtoms
            = ensemble_scattering->flags()
              & ENSEMBLE_SCATTERING_DUAL_V_ATOMS;
    const bool symmetricStorage
            = cphaseGateFlags & CPHASE_GATE_SYMMETRIC_STORAGE;

    eitParams.putPhasesInTheClassicalDrive = false;
    eitParams.counterpropagating = false;

    eitParams.input_electric_field_R_factor = 1;
    eitParams.inputE_width = sigmaInitial/v_g_eit;
    eitParams.inputE_mean = INPUT_E_MEAN_IN_UNITS_OF_WIDTH*eitParams.inputE_width;

    if (dualVAtoms && !symmetricStorage) {
        // For the Dual-V atoms, we can store regular
        // EIT wave packets (Gaussians).
        eitParams.input_electric_field_L_factor = 0;
    } else {
        // For the Lambda-type ensemble, we need to store
        // only in the atoms that are not on the nodes of
        // the classical drive. Hence, we send another pulse
        // from the left such that pulses from the right (with
        // the spatial phase exp(I*k*z)) and from the left (with
        // the spatial phase exp(-I*k*z)) interfere inside the
        // ensemble and produce the desired patters (that varies
        // as cos(k*z)). Also see "psi_analytical_EIT_storage"
        // arrays below.
        //
        // We can choose to do the same for the dual-V atoms too.
        eitParams.input_electric_field_L_factor = 1;
    }
    if (randomPlacement) {
        eitParams.randomAtomPositions = true;
    }

    const EvolutionMethod method
            = EvolutionMethod::Default;
    int flags = 0;
    flags |= LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_ELECTRIC_FIELD;

    // We are going to handle restarting ourselves here:
    //flags |= LAMBDA_HAMILTONIAN_EVOLVE_RESTART_IF_BOGUS_RESULTS;

    const double t_free_space = std::abs(eitParams.inputE_mean);
    const double t_storage = t_EIT + t_free_space;
    cphaseParameters->t_storage = t_storage;

    double normThreshold;
    double fidThreshold;
    double dtInitial;
    set_stored_spin_wave_set_tolerances(dtInitial, normThreshold,
                                        fidThreshold,
                                        onlyAnalyticalCalculation);

    std::vector<std::vector<double>> atomPositionsAllRealizations0;
    std::vector<std::vector<double>> atomPositionsAllRealizations1;
    std::vector<std::vector<std::complex<double>>> RAllRealizations;

    if (randomPlacement) {
        atomPositionsAllRealizations0.reserve(numRealizations);
        for (int n = 0; n < numRealizations; ++n) {
            const unsigned long int randomSeed
                    = cphaseParameters->randomSeeds[n];
            ensemble_scattering->setRandomSeed(randomSeed);

            const std::vector<double> atomPositions = ensemble_scattering->atomPositions();
            atomPositionsAllRealizations0.push_back(atomPositions);
        }
        atomPositionsAllRealizations1.reserve(numRealizations);
        for (int n = 0; n < numRealizations; ++n) {
            const unsigned long int randomSeed
                    = cphaseParameters->randomSeeds[n];
            // This will select a different ensemble realization
            // for the same randomSeed by discarding the first NAtoms
            // random numbers which determine the placement of
            // the atoms.
            ensemble_scattering->setRandomSeed(randomSeed, 1);

            const std::vector<double> atomPositions = ensemble_scattering->atomPositions();
            atomPositionsAllRealizations1.push_back(atomPositions);
            std::vector<std::complex<double>> R
                    = calculate_imp_reflection_coefficient_array(
                        cphaseParameters, cphaseGateFlags,
                        ensemble_scattering);
            RAllRealizations.push_back(R);
        }
    } else {
        atomPositionsAllRealizations0.reserve(numRealizations);
        RAllRealizations.reserve(numRealizations);
        for (int n = 0; n < numRealizations; ++n) {
            const unsigned long int randomSeed
                    = cphaseParameters->randomSeeds[n];
            ensemble_scattering->setRandomSeed(randomSeed);

            const std::vector<double> atomPositions = ensemble_scattering->atomPositions();
            atomPositionsAllRealizations0.push_back(atomPositions);
            std::vector<std::complex<double>> R
                    = calculate_imp_reflection_coefficient_array(
                        cphaseParameters, cphaseGateFlags,
                        ensemble_scattering);
            RAllRealizations.push_back(R);
        }
        atomPositionsAllRealizations1 = atomPositionsAllRealizations0;
    }
    const std::complex<double> noImpR = no_impurity_reflection_coefficient(
                cphaseParameters->delta,
                cphaseParameters->g1d,
                cphaseParameters->NAtoms,
                cphaseParameters->Deltac,
                cphaseParameters->OmegaScattering,
                cphaseParameters->kd_ensemble,
                cphaseParameters->kL1,
                cphaseParameters->kL2,
                ensemble_scattering,
                cphaseGateFlags);

#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "Doing EIT storage" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
    if (onlyAnalyticalCalculation) {
        const double storageTolAbs = STORED_SPIN_WAVE_TOLERANCE_ABS;
        const double storageTolRel = STORED_SPIN_WAVE_TOLERANCE_REL;
        std::vector<std::complex<double>> S_plus_array(NAtoms);
        std::vector<std::complex<double>> S_minus_array(NAtoms);
        if (dualVAtoms && !symmetricStorage) {
            if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_STORAGE_WITH_DISPERSION_RELATION) {
                S_plus_array = calculate_stored_spinwave_eit_dispersion_relation_plus(eitParams, true);
            } else {
                S_plus_array = calculate_stored_spinwave_eit_kernel_plus(eitParams, t_storage, storageTolAbs, storageTolRel, true);
            }
        } else {
            if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_STORAGE_WITH_DISPERSION_RELATION) {
                S_plus_array = calculate_stored_spinwave_eit_dispersion_relation_plus(eitParams, true);
                S_minus_array = calculate_stored_spinwave_eit_dispersion_relation_minus(eitParams, true);
            } else {
                S_plus_array = calculate_stored_spinwave_eit_kernel_plus(eitParams, t_storage, storageTolAbs, storageTolRel, true);
                S_minus_array = calculate_stored_spinwave_eit_kernel_minus(eitParams, t_storage, storageTolAbs, storageTolRel, true);
            }
        }
        for (int n = 0; n < numRealizations; ++n) {
            sol[n].psi_after_storage0 = Eigen::VectorXcd::Zero(2*NAtoms);
            sol[n].psi_after_storage_analytical_plus_ikz = Eigen::VectorXcd::Zero(NAtoms);
            sol[n].psi_after_storage_analytical_minus_ikz = Eigen::VectorXcd::Zero(NAtoms);

            // For the analytical shapes we need to do the same distinction
            // between Dual-V and Lambda-type. See the comment above.
            if (dualVAtoms && !symmetricStorage) {
                #pragma omp parallel for
                for (int i = 0; i < NAtoms; ++i) {

                    const double dz = 1.0/NAtoms;

                    if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_STORAGE_WITH_DISPERSION_RELATION) {
                        const std::complex<double> S_plus = S_plus_array[i];
                        sol[n].psi_after_storage0(i + NAtoms)
                                = S_plus*std::sqrt(dz);
                    } else {
                        const std::complex<double> S_plus = S_plus_array[i];

                        sol[n].psi_after_storage0(i + NAtoms) = S_plus*std::sqrt(dz);

                        if (std::isnan(sol[n].psi_after_storage0(i + NAtoms).real())) {
                            std::cout << "psi_analytical_EIT_storage[" << i << "] = "
                                      << sol[n].psi_after_storage0(i + NAtoms)
                                      << std::endl;
                        }
                    }
                // The +ikz and -ikz components of the stored spinwave.
                sol[n].psi_after_storage_analytical_plus_ikz(i)
                        = sol[n].psi_after_storage0(i + NAtoms);
                sol[n].psi_after_storage_analytical_minus_ikz(i) = 0;
                }
            } else {
                #pragma omp parallel for
                for (int i = 0; i < NAtoms; ++i) {
                    const double z = static_cast<double>(i)/NAtoms;
                    const double dz = 1.0/NAtoms;

                    std::complex<double> S_plus;
                    std::complex<double> S_minus;
                    if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_STORAGE_WITH_DISPERSION_RELATION) {
                        S_plus = S_plus_array[i];
                        S_minus = S_minus_array[i];
                        const double phase = kd_ensemble*M_PI*z*NAtoms;
                        const std::complex<double> phase_factor = std::exp(I*phase);
                        sol[n].psi_after_storage0(i + NAtoms)
                                = M_SQRT1_2
                                 *(S_plus
                                   *phase_factor
                                  +S_minus
                                   *std::conj(phase_factor))
                                *std::sqrt(dz);
                    } else {
                        S_plus = S_plus_array[i];
                        S_minus = S_minus_array[i];
                        const double phase = kd_ensemble*M_PI*z*NAtoms;
                        const std::complex<double> phase_factor = std::exp(I*phase);
                        sol[n].psi_after_storage0(i + NAtoms)
                                = M_SQRT1_2
                                 *(S_plus
                                   *phase_factor
                                  +S_minus
                                   *std::conj(phase_factor))
                                *std::sqrt(dz);
                        if (std::isnan(sol[n].psi_after_storage0(i + NAtoms).real())) {
                            std::cout << "psi_analytical_EIT_storage[" << i << "] = "
                                      << sol[n].psi_after_storage0(i + NAtoms)
                                      << std::endl;
                        }
                    }

                    // The +ikz and -ikz components of the stored spinwave.
                    sol[n].psi_after_storage_analytical_plus_ikz(i)
                            = S_plus*std::sqrt(dz);
                    sol[n].psi_after_storage_analytical_minus_ikz(i)
                            = S_minus*std::sqrt(dz);
                }
            }
            sol[n].psi_after_storage1 = sol[n].psi_after_storage0;
            sol[n].psi_after_scattering = sol[n].psi_after_storage0;
            sol[n].psi_after_scattering_analytical_plus_ikz = sol[n].psi_after_storage_analytical_plus_ikz;
            sol[n].psi_after_scattering_analytical_minus_ikz = sol[n].psi_after_storage_analytical_minus_ikz;

            const std::vector<std::complex<double>> R = RAllRealizations[n];
            #pragma omp parallel for
            for (int i = 0; i < NAtoms; ++i) {
                sol[n].psi_after_scattering(i + NAtoms) *= R[i];
                if (std::isnan(sol[n].psi_after_scattering(i + NAtoms).real())) {
                    std::cout << "psi_after_scattering[" << i << "] = "
                              << sol[n].psi_after_scattering(i + NAtoms)
                              << std::endl;
                }
            }
            if (!dualVAtoms) {
                assert(kd_ensemble == 0.5 && !randomPlacement
                       && "Only regular kd_ensemble=0.5 is supported here!");
                if (kd_ensemble != 0.5 || randomPlacement) {
                    std::cout << "Only regular kd_ensemble=0.5 is supported here!" << std::endl;
                }
                for (int i = 0; i < NAtoms; ++i) {
                    // The issue here is that +ikz and -ikz spinwave components need to be
                    // treated separately to be able to use Gorshkov's analytical
                    // expressions, but be summed, when multiplying the reflection coefficient.
                    // We will do here something that only works for kd=pi/2, where every other
                    // atom is on the antinode. Since exp(+ikz)+exp(-ikz)=2*cos(kz),
                    // when the reflection coefficient is multiplied onto the sum of the two
                    // components of the spinwave, the values of the reflection coefficient
                    // at odd multiples of pi/2 (odd i) will be effectively ignored by the spinwaves.
                    // On the other hand, when retrieving we need to look at the two components
                    // of the spinwave separately, even including those positions, where the two
                    // components will desctructively interfere (odd multiples of pi/2).
                    // What we do here is to multiply the values of the reflection coefficient that
                    // are  taken to be equal to the ones on the place just to the left of those
                    // positions with destructive interference.
                    sol[n].psi_after_scattering_analytical_plus_ikz(i) *= R[i - i%2];
                    sol[n].psi_after_scattering_analytical_minus_ikz(i) *= R[i - i%2];
                }
            }
            if (dualVAtoms) {
                // For the dual-V atoms we do not need to do any voodoo
                // dancing with shifting the reflection coefficient by one
                // because it is a smooth function contrary to the Lambda-type
                // case.
                for (int i = 0; i < NAtoms; ++i) {
                    sol[n].psi_after_scattering_analytical_plus_ikz(i) *= R[i];
                    sol[n].psi_after_scattering_analytical_minus_ikz(i) *= R[i];
                }
            }
        }
    } else {
        for (int n = 0; n < numRealizations; ++n) {
            eitParams.atom_positions = atomPositionsAllRealizations0[n];
            LambdaHamiltonian1Excitation H_EIT;
            H_EIT.setParams(eitParams);
            sol[n].psi_after_storage0
                    = calculate_stored_spinwave_eit_field_elimination(
                        H_EIT, method, t_storage, dtInitial, normThreshold,
                        flags);
            if (randomPlacement) {
                // For randomly placed atoms, the ensemble
                // that is scattered from will, in general, have
                // different positions of the atoms from
                // the ensemble that we only use as a memory
                // (we need two ensembles for the dual-rail encoding
                // of photonic qubits)
                eitParams.atom_positions = atomPositionsAllRealizations1[n];
                H_EIT.setParams(eitParams);
                sol[n].psi_after_storage1
                        = calculate_stored_spinwave_eit_field_elimination(
                            H_EIT, method, t_storage, dtInitial, normThreshold,
                            flags);
            } else {
                sol[n].psi_after_storage1 = sol[n].psi_after_storage0;
            }
            sol[n].psi_after_scattering = sol[n].psi_after_storage1;
            const std::vector<std::complex<double>> R = RAllRealizations[n];
            #pragma omp parallel for
            for (int i = 0; i < NAtoms; ++i) {
                sol[n].psi_after_scattering(i + NAtoms) *= R[i];
                if (std::isnan(sol[n].psi_after_scattering(i + NAtoms).real())) {
                    std::cout << "psi_after_scattering[" << i << "] = "
                              << sol[n].psi_after_scattering(i + NAtoms)
                              << std::endl;
                }
            }
        }
    }
    if (cphaseGateFlags & CPHASE_GATE_PROJECT_SPINWAVES) {
        return;
    }

#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "Doing EIT retrieval" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT

    const double t_retrieval = timeToPassEnsemble;
    cphaseParameters->t_retrieval = t_retrieval;
    eitParams.input_electric_field_R_factor = 0;
    eitParams.input_electric_field_L_factor = 0;
    retrieve_stored_spinwave(
                sol, cphaseParameters,
                onlyAnalyticalCalculation, dualVAtoms, symmetricStorage,
                randomPlacement,
                atomPositionsAllRealizations0,
                atomPositionsAllRealizations1,
                noImpR, quadK_r, eitParams, dtInitial,
                t_retrieval, method, normThreshold, fidThreshold, flags,
                numRealizations);
}

struct find_optimal_tNo_interaction_params
{
    const SpinwaveAndFieldVector &sol;
    const std::vector<std::complex<double>> &noImpR;
    int cphaseGateFlags;
    bool onlyAnalyticalCalculation;
    find_optimal_tNo_interaction_params(
            const SpinwaveAndFieldVector &sol_a,
            const std::vector<std::complex<double>> &noImpR_a) :
        sol(sol_a),
        noImpR(noImpR_a)
    {}
};

double find_optimal_tNo_interaction_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_tNo_interaction_params *p
            = (find_optimal_tNo_interaction_params *) params;
    CphaseFidelities fid;
    const double rho = x[0];
    const double theta = x[1];
    const std::complex<double> I(0,1);
    const std::complex<double> tNoInteraction = rho*std::exp(I*theta);
    calculate_cphase_fidelities_for_tNoInteraction(p->sol,
                                                   &fid,
                                                   p->noImpR, tNoInteraction,
                                                   p->cphaseGateFlags,
                                                   p->onlyAnalyticalCalculation);
    return -fid.F_swap;
}

double find_optimal_tNoInteraction(double *rho, double *theta,
                                   const SpinwaveAndFieldVector &sol,
                                   const std::vector<std::complex<double>> &noImpR,
                                   int cphaseGateFlags,
                                   bool onlyAnalyticalCalculation,
                                   bool globalOptimization,
                                   double xtol_abs)
{
    const int numParams = 2;
    find_optimal_tNo_interaction_params params(sol, noImpR);
    params.cphaseGateFlags = cphaseGateFlags;
    params.onlyAnalyticalCalculation = onlyAnalyticalCalculation;
    nlopt::algorithm algorithm;
    if (globalOptimization) {
        algorithm = nlopt::GN_DIRECT_L;
    } else {
        algorithm = nlopt::LN_SBPLX;
    }
    nlopt::opt opt(algorithm, numParams);
    std::vector<double> x(numParams);
    x[0] = *rho;
    x[1] = *theta;
    std::vector<double> step(numParams);
    step[0] = 0.01;
    step[1] = 0.001;
    opt.set_initial_step(step);

    std::vector<double> lb(numParams);
    lb[0] = 0;
    lb[1] = -M_PI;
    opt.set_lower_bounds(lb);

    std::vector<double> ub(numParams);
    ub[0] = 1;
    ub[1] = M_PI;
    opt.set_upper_bounds(ub);

    opt.set_min_objective(find_optimal_tNo_interaction_f, &params);
    opt.set_xtol_abs(xtol_abs);

    double minf;
    nlopt::result result = opt.optimize(x, minf);
    *rho = x[0];
    *theta = x[1];
    return -minf;
}

double find_optimal_Deltac_sigma_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_Deltac_sigma_params *p
            = (find_optimal_Deltac_sigma_params *) params;

    const double sigma = x[0];
    const double Deltac = x[1];

    find_delta_at_first_resonance_params firstResonanceParams;
    firstResonanceParams.Deltac = Deltac;
    // Use the regularly placed atomic ensemble to
    // find the resonance detuning
    firstResonanceParams.f = p->ensemble_scattering_regular;
    // kd_ensemble_regular is set to some incommensurate
    // value (0.266).
    firstResonanceParams.kd_ensemble = p->kd_ensemble_regular;
    firstResonanceParams.g1d = p->cphaseParameters->g1d;
    firstResonanceParams.NAtoms = p->cphaseParameters->NAtoms;
    firstResonanceParams.OmegaScattering = p->cphaseParameters->OmegaScattering;
    if (!find_delta_at_first_resonance(&firstResonanceParams)) {
        std::cout << "Deltac = " << Deltac << ", "
                  << "Couldn't find Delta at first resonance!" << std::endl;
        assert(0 && "Couldn't find Delta at first resonance!");
        return 0;
    }

    p->cphaseParameters->delta = firstResonanceParams.delta;

    const double sigma_old = p->cphaseParameters->sigma;
    const double Deltac_old = p->cphaseParameters->Deltac;
    p->cphaseParameters->sigma = sigma;
    p->cphaseParameters->Deltac = Deltac;
    const double sigmaDiff = std::abs(sigma_old - sigma);
    const double DeltacDiff = std::abs(Deltac_old - Deltac);

    const double F_CJ_cond_old = p->fid_tNoInteraction_one->F_CJ_conditional;
    calculate_cphase_fidelities_numerical_wrapper(p);
    const double condFidTol = 1e-7;
    const double sigmaTol = 1e-7;
    const double DeltacTol = 1e-7;
    const double condFidDiff = std::abs(F_CJ_cond_old-p->fid_tNoInteraction_one->F_CJ_conditional);
    if (p->stopWhenCondFidTolReached && condFidDiff < condFidTol && sigmaDiff < sigmaTol && DeltacDiff < DeltacTol) {
        throw nlopt::forced_stop();
    }

#ifdef CPHASE_GATE_DEBUG_PRINT
    //double rho = 1;
    //double theta = 0;
    //find_optimal_tNoInteraction(&rho, &theta, p->sol, p->noImpR,
    //                            p->cphaseGateFlags,
    //                            p->onlyAnalyticalCalculation);
    //CphaseFidelities fid;
    //const std::complex<double> I(0,1);
    //const std::complex<double> tNoInteraction = rho*std::exp(I*theta);
    //calculate_cphase_fidelities_for_tNoInteraction(p->sol,
    //                                               &fid,
    //                                               p->noImpR, tNoInteraction,
    //                                               p->cphaseGateFlags,
    //                                               p->onlyAnalyticalCalculation);
    std::cout << "g1d = " << p->cphaseParameters->g1d
              << ", N = " << p->cphaseParameters->NAtoms
              << ", sigma = " << sigma
              << ", Deltac = " << Deltac
              << ", delta = " << p->cphaseParameters->delta
              << ", F_CJ_tNoInteraction_one = " << p->fid_tNoInteraction_one->F_CJ
              << ", F_CJ_cond_tNoInteraction_one = " << p->fid_tNoInteraction_one->F_CJ_conditional
    //          << ", F_CJ = " << fid.F_CJ
    //          << ", F_CJ_cond = " << fid.F_CJ_conditional
              << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
    return -p->fid_tNoInteraction_one->F_CJ;
}

double find_optimal_Deltac_f(unsigned n, const double *x, double *grad, void *params)
{
    find_optimal_Deltac_sigma_params *p
            = (find_optimal_Deltac_sigma_params *) params;

    const double Deltac = x[0];

    find_delta_at_first_resonance_params firstResonanceParams;
    firstResonanceParams.Deltac = Deltac;
    // Use the regularly placed atomic ensemble to
    // find the resonance detuning
    firstResonanceParams.f = p->ensemble_scattering_regular;
    // kd_ensemble_regular is set to some incommensurate
    // value (0.266).
    firstResonanceParams.kd_ensemble = p->kd_ensemble_regular;
    firstResonanceParams.g1d = p->cphaseParameters->g1d;
    firstResonanceParams.NAtoms = p->cphaseParameters->NAtoms;
    firstResonanceParams.OmegaScattering = p->cphaseParameters->OmegaScattering;
    if (!find_delta_at_first_resonance(&firstResonanceParams)) {
        std::cout << "Deltac = " << Deltac << ", "
                  << "Couldn't find Delta at first resonance!" << std::endl;
        assert(0 && "Couldn't find Delta at first resonance!");
        return 0;
    }

    p->cphaseParameters->delta = firstResonanceParams.delta;

    p->cphaseParameters->Deltac = Deltac;

    calculate_cphase_fidelities_numerical_wrapper(p);

    return -p->fid_tNoInteraction_one->F_CJ;
}

double find_Deltac_critical(find_optimal_Deltac_sigma_params *params, double Deltac_guess)
{
    const double Deltac_critical_err = 1e-8;
    double Deltac_critical = find_root_secant<double>([=] (double Deltac) -> double {
        return approximate_resonance_solution1_discriminant(
                    params->cphaseParameters->g1d, params->cphaseParameters->NAtoms,
                    Deltac,
                    params->cphaseParameters->OmegaScattering);
    }, Deltac_guess, 2*Deltac_guess, Deltac_critical_err);
    Deltac_critical += Deltac_critical_err;
    return Deltac_critical;
}

void guess_Deltac_sigma(find_optimal_Deltac_sigma_params *params, double &Deltac_critical)
{
    // Here we try to pick a reasonable starting value of Deltac
    // It is complicated by the fact that the cavity resonances
    // appear only for a finite range of Deltac and hence the code
    // that blindly tries to use a too big or too small value of
    // Deltac to find Delta, can return a NaN and nothing will work.

    double Deltac_guess = -100;
    bool found_Deltac_guess = false;
    for (int i = 0; i < 10; ++i) {
        find_delta_at_first_resonance_params firstResonanceParams;
        firstResonanceParams.Deltac = Deltac_guess;
        firstResonanceParams.f = params->ensemble_scattering;
        firstResonanceParams.g1d = params->cphaseParameters->g1d;
        firstResonanceParams.kd_ensemble = params->cphaseParameters->kd_ensemble;
        firstResonanceParams.NAtoms = params->cphaseParameters->NAtoms;
        firstResonanceParams.OmegaScattering = params->cphaseParameters->OmegaScattering;
        if (find_delta_at_first_resonance(&firstResonanceParams)) {
            found_Deltac_guess = true;
            break;
        }
        Deltac_guess /= 2;
    }
    if (!found_Deltac_guess) {
        std::cout << "Couldn't find starting Deltac!" << std::endl;
        assert(0 && "Couldn't find starting Deltac!");
    }
    Deltac_critical = find_Deltac_critical(params, Deltac_guess);
#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "Deltac_critical = " << Deltac_critical << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
    Deltac_guess = Deltac_critical/2;

    double Deltac_for_optimal_fidelity_guess;
    if (params->cphaseGateFlags & CPHASE_GATE_SAGNAC_SCATTERING) {
        Deltac_for_optimal_fidelity_guess = -params->cphaseParameters->g1d
                  *std::pow(params->cphaseParameters->NAtoms, 3.0/4)
                  /std::sqrt(8*M_PI);
    } else {
        Deltac_for_optimal_fidelity_guess = -params->cphaseParameters->g1d
                  *std::pow(params->cphaseParameters->NAtoms, 3.0/4)
                  /std::sqrt(4*M_PI);
    }
    if (Deltac_guess < Deltac_for_optimal_fidelity_guess) {
        Deltac_guess = Deltac_for_optimal_fidelity_guess;
    }
    const int periodLength
            = find_period_length_from_kd(params->cphaseParameters->kd_ensemble);
    const bool disorderedEnsemble = (periodLength < 0)
            || (params->cphaseGateFlags & CPHASE_GATE_RANDOM_ATOM_POSITIONS);
    const bool dualVatoms = params->ensemble_scattering->flags()
                            & ENSEMBLE_SCATTERING_DUAL_V_ATOMS;
    if (disorderedEnsemble && !dualVatoms && Deltac_guess < -50) {
        // Deltac_guess above uses the
        // equations for the regular placement of
        // atoms with kd=pi/2. For the randomly placed atoms
        // these formulas give a much to big (in absolute value)
        // Deltac
        Deltac_guess = -50;
    }
#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "Deltac_guess = " << Deltac_guess << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
    params->cphaseParameters->Deltac = Deltac_guess;

    // A guess for sigma
    params->cphaseParameters->sigma = 0.15;
}

void calculate_cphase_fidelities_numerical_for_optimal_Deltac_sigma(
        CphaseFidelities *fid, CphaseFidelities *fid_tNoInteraction_one,
        SpinwaveAndFieldVector &sol,
        std::vector<CphaseDiagnosticData> &diagData,
        CphaseGateFidelityParameters *cphaseParameters, int cphaseGateFlags,
        EnsembleScattering *ensemble_scattering)
{
    int numRandomSeeds = cphaseParameters->randomSeeds.size();
    find_optimal_Deltac_sigma_params params;
    // Even if the atoms are placed randomly we can still
    // choose to do the optimization with a regular ensemble.
    const bool randomPlacement
            = ensemble_scattering->flags()
              & ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS;
    const bool regularOptimization
            = cphaseGateFlags
              & CPHASE_GATE_RANDOM_POSITIONS_OPTIMIZE_WITH_REGULAR;
    std::vector<unsigned long int> originalRandomSeeds;
    int regular_flags = ensemble_scattering->flags();
    regular_flags &= ~ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS;

    // We always create the regularly spaced version of
    // the ensemble, which we use to find the resonance
    // detunings.
    params.ensemble_scattering_regular
            = new EnsembleScattering(regular_flags);
    if (randomPlacement && cphaseParameters->kd_ensemble == 0.5) {
        // Put some incommensurate interatomic spacing
        params.kd_ensemble_regular = 0.266;
    } else {
        params.kd_ensemble_regular = cphaseParameters->kd_ensemble;
    }
    if (randomPlacement && regularOptimization) {
        params.ensemble_scattering = params.ensemble_scattering_regular;
        CphaseGateFidelityParameters *regularCphaseParameters
                = new CphaseGateFidelityParameters;
        *regularCphaseParameters = *cphaseParameters;
        if (regularCphaseParameters->kd_ensemble == 0.5) {
            regularCphaseParameters->kd_ensemble = params.kd_ensemble_regular;
        }
        params.cphaseParameters = regularCphaseParameters;
        // Even if we are requested to calculate the fidelities
        // for the randomly placed ensembles, we first do the
        // optimization with a regularly spaced ensemble. In
        // which case there should only be one realization of
        // the ensemble that we need to consider.
        // We also replace the random seeds array by
        // another one with a single element, keeping
        // the original copy for the case where we need
        // to compute the final result with a randomly
        // placed ensemble.
        originalRandomSeeds = cphaseParameters->randomSeeds;
        numRandomSeeds = 1;
        params.cphaseParameters->randomSeeds = std::vector<unsigned long int>(1);
    } else {
        params.ensemble_scattering = ensemble_scattering;
        params.cphaseParameters = cphaseParameters;
        if (!randomPlacement) {
            numRandomSeeds = 1;
            params.cphaseParameters->randomSeeds = std::vector<unsigned long int>(1);
        }
    }
    params.sol = SpinwaveAndFieldVector(numRandomSeeds);
    params.noImpR.resize(numRandomSeeds);

    const bool analyticalOptimization
            = cphaseGateFlags & CPHASE_GATE_ANALYTICAL_OPTIMIZATION;
    const bool analyticalFinalResult
            = cphaseGateFlags & CPHASE_GATE_ANALYTICAL_FINAL_RESULT;

    if (analyticalOptimization) {
        params.onlyAnalyticalCalculation = true;
    } else {
        params.onlyAnalyticalCalculation = false;
    }
    params.stopWhenCondFidTolReached = true;

    const int numParams = 2;
    params.fid_tNoInteraction_one = fid_tNoInteraction_one;
    params.cphaseGateFlags = cphaseGateFlags;

    double Deltac_critical = -HUGE_VAL;
    if (params.cphaseParameters->Deltac == 0 || params.cphaseParameters->sigma == 0) {
        guess_Deltac_sigma(&params, Deltac_critical);
    } else {
        Deltac_critical = find_Deltac_critical(&params, params.cphaseParameters->Deltac);
    }

    std::vector<double> x(numParams);
    x[0] = params.cphaseParameters->sigma;
    x[1] = params.cphaseParameters->Deltac;
    std::vector<double> step(numParams);
    step[0] = 0.01;
    step[1] = std::max(params.cphaseParameters->Deltac/20, 0.1);

    nlopt::algorithm algo = nlopt::LN_SBPLX;

    nlopt::opt opt(algo, numParams);
    opt.set_initial_step(step);

    std::vector<double> lb(numParams);
    lb[0] = 0.0005; // Should be a reasonable lower bound for sigma
    lb[1] = Deltac_critical;
    opt.set_lower_bounds(lb);

    std::vector<double> ub(numParams);
    ub[0] = 1; // sigma > L is not going to work well anyway
    ub[1] = 0;
    opt.set_upper_bounds(ub);

    opt.set_min_objective(find_optimal_Deltac_sigma_f, &params);

    set_optmization_tolerance(opt, params.onlyAnalyticalCalculation);

#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "Doing local optimization" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
    double minf;
    try {
        nlopt::result result = opt.optimize(x, minf);
    } catch (nlopt::forced_stop) {
#ifdef CPHASE_GATE_DEBUG_PRINT
        std::cout << "Early stop because the conditional fidelity tolerance was satisfied" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
    } catch (...) {
        throw;
    }

#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "   Finished optimization" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT

    // Whether to do one last call to the
    // fidelity function. It is needed in the
    // case when the parameters of the optimization
    // are different from the parameters of the
    // desired result. See below.
    bool oneMoreCallOfFidelityFunction = false;

    // If the optimization is analytical but the final
    // result is numerical, then we need to compute the final
    // result with the "onlyAnalyticalCalculation" set
    // to false.
    if (analyticalOptimization && !analyticalFinalResult) {
        params.onlyAnalyticalCalculation = false;
        oneMoreCallOfFidelityFunction = true;
    }
    // If the optimization is numerical but the final
    // result is analytical, then we need to compute the final
    // result with the "onlyAnalyticalCalculation" set
    // to true.
    if (!analyticalOptimization && analyticalFinalResult) {
        params.onlyAnalyticalCalculation = true;
        oneMoreCallOfFidelityFunction = true;
    }
    if (randomPlacement && regularOptimization) {
        // Since we did the optimization with the
        // regularly spaced ensemble above, switch to the
        // actual randomly placed ensemble here and use
        // the optimal parameters (Deltac and sigma) obtained above.

        // Copy the optimal values from the temporary
        // parameters struct before deleting it.
        cphaseParameters->delta = params.cphaseParameters->delta;
        cphaseParameters->sigma = params.cphaseParameters->sigma;
        cphaseParameters->Deltac = params.cphaseParameters->Deltac;
        delete params.cphaseParameters;

        params.ensemble_scattering = ensemble_scattering;
        params.cphaseParameters = cphaseParameters;
        params.cphaseParameters->randomSeeds = originalRandomSeeds;
        numRandomSeeds = cphaseParameters->randomSeeds.size();
        if (sol.size() != numRandomSeeds) {
            sol.resize(numRandomSeeds);
        }
        params.sol = sol;
        params.noImpR = std::vector<std::complex<double>>(numRandomSeeds);

        // Do another round of optimizing Deltac here.
        // This is should be a less expensive
        // than doing optimization over both Deltac and sigma
        // with the randomly placed ensemble from the beginning.
        // This way, we have optimized Deltac and sigma with
        // the regularly placed ensemble above, and now only
        // optimize Deltac without touching the value of sigma.
        nlopt::opt optDeltac(algo, 1);
        std::vector<double> xDeltac(1);
        xDeltac[0] = cphaseParameters->Deltac;
        std::vector<double> stepDeltac(1);
        stepDeltac[0] = std::max(cphaseParameters->Deltac/20, 0.1);
        optDeltac.set_initial_step(stepDeltac);

        std::vector<double> lbDeltac(1);
        lbDeltac[0] = Deltac_critical;
        optDeltac.set_lower_bounds(lbDeltac);

        std::vector<double> ubDeltac(1);
        ubDeltac[0] = 0;
        optDeltac.set_upper_bounds(ubDeltac);

        optDeltac.set_min_objective(find_optimal_Deltac_f, &params);

        set_optmization_tolerance(optDeltac, params.onlyAnalyticalCalculation);
        double minf;
        nlopt::result result = optDeltac.optimize(xDeltac, minf);

        oneMoreCallOfFidelityFunction = true;
    }
    if (oneMoreCallOfFidelityFunction) {
        //find_optimal_Deltac_sigma_f(2, x.data(), nullptr, &params);
        calculate_cphase_fidelities_numerical_wrapper(&params);
    }
    delete params.ensemble_scattering_regular;

    // The optimization was performed over a copy
    // of this array. Copy the data back to the reference
    // in the argument of this function.
    sol = params.sol;

    // Find the optimal value of the beam splitter
    // transmission coefficient.

    // Starting values
    double rho = 0.9;
    double theta = 0;
    if (params.cphaseParameters->kL1 != 0 || params.cphaseParameters->kL1 != 0) {
        // Theta can be significantly different from 0
        // in this case. Therefore do three different local
        // optimazations with starting theta: -pi/2, 0, pi/2.
        // Choose the one that gives the biggetst F_CJ_cond.
        double rho_old;
        double theta_old;
        double F_CJ_cond_old;
        double F_CJ_cond;
        // Starting values
        rho = 0.9;
        theta = -M_PI/2;
        F_CJ_cond_old = find_optimal_tNoInteraction(
                    &rho, &theta, sol, params.noImpR, params.cphaseGateFlags,
                    params.onlyAnalyticalCalculation);
        rho_old = rho;
        theta_old = theta;

        // Starting values
        rho = 0.9;
        theta = 0;
        F_CJ_cond = find_optimal_tNoInteraction(
                    &rho, &theta, sol, params.noImpR, params.cphaseGateFlags,
                    params.onlyAnalyticalCalculation);
        if (F_CJ_cond > F_CJ_cond_old) {
            rho_old = rho;
            theta_old = theta;
            F_CJ_cond_old = F_CJ_cond;
        }

        // Starting values
        rho = 0.9;
        theta = M_PI/2;
        F_CJ_cond = find_optimal_tNoInteraction(
                    &rho, &theta, sol, params.noImpR, params.cphaseGateFlags,
                    params.onlyAnalyticalCalculation);
        if (F_CJ_cond > F_CJ_cond_old) {
            rho_old = rho;
            theta_old = theta;
            F_CJ_cond_old = F_CJ_cond;
        }
        rho = rho_old;
        theta = theta_old;
    } else {
        find_optimal_tNoInteraction(&rho, &theta, sol, params.noImpR,
                                    params.cphaseGateFlags,
                                    params.onlyAnalyticalCalculation);
    }
#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "   Finished finding optimal tNoInteraction" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
    cphaseParameters->tNoInteractionAbs = rho;
    cphaseParameters->tNoInteractionArg = theta;
    const std::complex<double> I(0,1);
    const std::complex<double> tNoInteraction = rho*std::exp(I*theta);
    calculate_cphase_fidelities_for_tNoInteraction(sol,
                                                   fid,
                                                   params.noImpR,
                                                   tNoInteraction,
                                                   cphaseGateFlags,
                                                   params.onlyAnalyticalCalculation);
#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "   Finished computing one more fidelity calculation" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT

    for (int i = 0; i < numRandomSeeds; ++i) {
        const unsigned long int randomSeed
                = params.cphaseParameters->randomSeeds[i];
        ensemble_scattering->setRandomSeed(randomSeed);
        sol[i].zValsWithoutScattering = Eigen::ArrayXd::Zero(params.cphaseParameters->NAtoms);
        for (int j = 0; j < params.cphaseParameters->NAtoms; ++j) {
            sol[i].zValsWithoutScattering(j) = ensemble_scattering->atomPositions()[j];
        }
        // The ensemble with scattering will in general have
        // different positions of the atoms (if random placement
        // is assumed).
        ensemble_scattering->setRandomSeed(randomSeed, 1);
        sol[i].zValsWithScattering = Eigen::ArrayXd::Zero(params.cphaseParameters->NAtoms);
        for (int j = 0; j < params.cphaseParameters->NAtoms; ++j) {
            sol[i].zValsWithScattering(j) = ensemble_scattering->atomPositions()[j];
        }
    }
    if (diagData.size() != numRandomSeeds) {
        diagData.resize(numRandomSeeds);
    }

    // Calculate the different overlaps. These are the
    // same overlaps that are needed for the fidelity calculations
    // in
    // calculate_cphase_fidelities_E_field
    // and
    // calculate_cphase_fidelities_E_field_with_weights
    // functions. The idea is that this "diagnostic data"
    // could be used to reconstruct the CJ fidelity with
    // arbitrary tNoInteraction.

    // Take the reference outgoing electric field mode
    // to be the average of the modes from each realization
    const int num_t_points = sol.tVals.size();
    Eigen::VectorXcd E_without_scattering_avg = Eigen::VectorXcd::Zero(num_t_points);
    for (int i = 0; i < numRandomSeeds; ++i) {
        E_without_scattering_avg += sol[i].E_without_scattering0;
    }
    E_without_scattering_avg /= numRandomSeeds;
    double E_without_scattering_avg_squaredNorm;
    if (analyticalFinalResult) {
        E_without_scattering_avg_squaredNorm = squaredNorm_with_weights(E_without_scattering_avg, sol.tWeights);
    } else {
        E_without_scattering_avg_squaredNorm = E_without_scattering_avg.squaredNorm();
    }
    fid->single_photon_storage_retrieval_eff = E_without_scattering_avg_squaredNorm;

    for (int i = 0; i < numRandomSeeds; ++i) {
        diagData[i].noImpurityR = params.noImpR[i];
        diagData[i].spinwave_after_storage_norm = sol[i].psi_after_storage0.norm();
        diagData[i].spinwave_after_scattering_norm = sol[i].psi_after_scattering.norm();
        diagData[i].impurityRfromSpinwave = sol[i].psi_after_storage0.dot(sol[i].psi_after_scattering)
                                          /sol[i].psi_after_storage0.squaredNorm();
        if (analyticalFinalResult) {
            // If the final result is analytical, we have been using
            // the tanh-sinh quadrature, and the norms and overlaps
            // have to be computed taking into account the weights.
            const double E_without_scattering_squaredNorm = squaredNorm_with_weights(sol[i].E_without_scattering0, sol.tWeights);
            diagData[i].E_without_scattering_norm = std::sqrt(E_without_scattering_squaredNorm);
            diagData[i].E_with_scattering_norm = std::sqrt(squaredNorm_with_weights(sol[i].E_with_scattering, sol.tWeights));
            if (diagData[i].E_without_scattering_norm != 0) {
                // We can choose not to compute the electric field
                // and then its norm will be equal to zero resulting
                // here in a NaN. In principle it's harmless, and just
                // gets stored as "nan" in the data files. So this check
                // is more cosmetic than anything.
                diagData[i].impurityRfromE
                        = inner_product_with_weights(E_without_scattering_avg,
                                                     sol[i].E_with_scattering,
                                                     sol.tWeights)
                          /E_without_scattering_avg_squaredNorm;
                diagData[i].E_without_scattering_avg_overlap
                        = inner_product_with_weights(E_without_scattering_avg,
                                                     sol[i].E_without_scattering0,
                                                     sol.tWeights)
                          /E_without_scattering_avg_squaredNorm;
            } else {
                diagData[i].impurityRfromE = 0;
            }
        } else {
            diagData[i].E_without_scattering_norm = sol[i].E_without_scattering0.norm();
            diagData[i].E_with_scattering_norm = sol[i].E_with_scattering.norm();
            if (diagData[i].E_without_scattering_norm != 0) {
                // We can choose not to compute the electric field
                // and then its norm will be equal to zero resulting
                // here in a NaN. In principle it's harmless, and just
                // gets stored as "nan" in the data files. So this check
                // is more cosmetic than anything.
                diagData[i].impurityRfromE
                        = E_without_scattering_avg.dot(
                            sol[i].E_with_scattering)
                          /E_without_scattering_avg_squaredNorm;
                diagData[i].E_without_scattering_avg_overlap
                        = E_without_scattering_avg.dot(
                            sol[i].E_without_scattering0)
                          /E_without_scattering_avg_squaredNorm;
            } else {
                diagData[i].impurityRfromE = 0;
            }
        }
    }
#ifdef CPHASE_GATE_DEBUG_PRINT
    std::cout << "   Finished computing diagnostic data" << std::endl;
#endif // CPHASE_GATE_DEBUG_PRINT
}
