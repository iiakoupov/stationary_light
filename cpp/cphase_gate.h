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

#ifndef CPHASE_GATE_H
#define CPHASE_GATE_H

#include <vector>
#include <complex>
#include "Eigen/Dense"
#include "threaded_eigen_matrix.h"
#include "lambda_hamiltonian.h"

#define CPHASE_GATE_RANDOM_ATOM_POSITIONS                          (1 << 0)
#define CPHASE_GATE_PROJECT_SPINWAVES                              (1 << 1)
#define CPHASE_GATE_ANALYTICAL_FINAL_RESULT                        (1 << 2)
#define CPHASE_GATE_ANALYTICAL_OPTIMIZATION                        (1 << 3)

// Whether to use the storage kernel from [Phys. Rev. A 76, 033805]
// to calculate the analytical stored spinwave. If set,
// the EIT dispersion relation is used instead.
#define CPHASE_GATE_ANALYTICAL_STORAGE_WITH_DISPERSION_RELATION    (1 << 4)

// Whether to do the EIT storage by splitting the incident
// pulse on a 50:50 beam splitter, so that it enters the ensemble
// symmetrically from both sides. This is the only way to do storage
// for the Lambda-type atoms, and hence this setting has no effect there.
// For the dual-V atoms, one can also choose to do storage from one
// side. Setting this flag makes the dual-V storage work like the
// Lambda-type atoms.
#define CPHASE_GATE_SYMMETRIC_STORAGE                              (1 << 5)

// Whether to assume that the ensemble was placed
// in a Sagnac interferometer. If this flag is *not* set,
// we instead assume the setup:
//      ----------     |\
// ----| Ensemble |----|\ <- Mirror
//      ----------     |\
// I.e. the ensemble acts as a mirror in a one-sided
// cavity.
#define CPHASE_GATE_SAGNAC_SCATTERING                              (1 << 6)

// If the atoms are positioned randomly, we can find the
// optimal Deltac and sigma with regularly placed atoms
// and then use the optimal Deltac and sigma to do the final
// evaluation of the fidelities and sucess probability
// with the actual randomly placed ensemble.
#define CPHASE_GATE_RANDOM_POSITIONS_OPTIMIZE_WITH_REGULAR         (1 << 7)

// Instead of having a fixed OmegaScattering and OmegaStorageRetrieval,
// vary them such that scattering time is constant.
#define CPHASE_GATE_VARY_OMEGA_SUCH_THAT_SCATTERING_TIME_IS_CONST  (1 << 8)

// This sets, how far away the input Gaussian pulse
// starts from the ensemble in units of its width.
// It does introduce and error to the input pulse, since the
// tail of the Gaussian is effectively cut off this way.
// Note that for big optical depths, the optimal width will
// be smaller and thus the effect of the truncation here will
// become relatively larger. In that case one can try increasing
// this value.
#define INPUT_E_MEAN_IN_UNITS_OF_WIDTH 4

// Absolute tolerance for the quadrature (numerical integration)
// when computing the stored spin wave using the EIT
// storage kernel
#define STORED_SPIN_WAVE_TOLERANCE_ABS 1e-10

// Relative tolerance for the quadrature (numerical integration)
// when computing the stored spin wave using the EIT
// storage kernel.
// It is set here to a rather loose value (10 %)
// since it will be checked in addition to the
// absolute tolerance above. It was added mostly to cope
// with the situations, when the computed value of
// the integral is an extremely small number with
// a matching extremly small absolute tolerance, which made
// the integration loop to erroneously assume that
// it should stop at this point. However, in those cases
// continuing with higher quadrature degrees eventually
// produced the correct answer. Therefore to not bail out
// prematurely, the relative tolerance check was added.
// The idea is that even if the absolute tolerance in
// those cases was below user supplied tolerance (sometimes
// even below machine tolerance), the relative tolerance was
// about 100 %. By requiring the relative tolerance to be 10 %,
// these premature integration loop exits were fixed.
#define STORED_SPIN_WAVE_TOLERANCE_REL 0.1

class EnsembleScattering;

std::complex<double> impurity_reflection_coefficient_discrete(
        int n, double delta, double g1d, int NAtoms, double Deltac,
        double Omega, double kd, double kL1, double kL2,
        EnsembleScattering *ensemble_scattering,
        int cphaseGateFlags);

std::complex<double> no_impurity_reflection_coefficient(
        double delta, double g1d, int NAtoms, double Deltac,
        double Omega, double kd, double kL1, double kL2,
        EnsembleScattering *ensemble_scattering,
        int cphaseGateFlags);

struct find_delta_at_first_resonance_params
{
    EnsembleScattering *f;
    int NAtoms;
    double g1d;
    double delta;
    double Deltac;
    double kd_ensemble;
    double OmegaScattering;
};

bool find_delta_at_first_resonance(find_delta_at_first_resonance_params *params);

struct SpinwaveAndField {
    Eigen::ArrayXd zValsWithoutScattering;
    Eigen::ArrayXd zValsWithScattering;
    Eigen::VectorXcd psi_after_storage0;
    Eigen::VectorXcd psi_after_storage1;
    Eigen::VectorXcd psi_after_scattering;
    Eigen::VectorXcd psi_after_storage_analytical_plus_ikz;
    Eigen::VectorXcd psi_after_storage_analytical_minus_ikz;
    Eigen::VectorXcd psi_after_scattering_analytical_plus_ikz;
    Eigen::VectorXcd psi_after_scattering_analytical_minus_ikz;
    Eigen::VectorXcd psi_after_retrieval_with_scattering;
    Eigen::VectorXcd psi_after_retrieval_without_scattering;

    Eigen::VectorXcd E_R_without_scattering0;
    Eigen::VectorXcd E_L_without_scattering0;
    Eigen::VectorXcd E_without_scattering0;

    Eigen::VectorXcd E_without_scattering1;

    Eigen::VectorXcd E_R_with_scattering;
    Eigen::VectorXcd E_L_with_scattering;
    Eigen::VectorXcd E_with_scattering;
};

struct SpinwaveAndFieldVector
{
    std::vector<SpinwaveAndField> vec;
    Eigen::ArrayXd tVals;
    Eigen::ArrayXd tWeights;
    SpinwaveAndFieldVector() = default;
    explicit SpinwaveAndFieldVector(int size) : vec(size) {}
    SpinwaveAndField &operator[](int i)
    {
        return vec[i];
    }
    const SpinwaveAndField &operator[](int i) const
    {
        return vec[i];
    }
    int size() const
    {
        return vec.size();
    }
    void resize(int size)
    {
        vec.resize(size);
    }
};

struct CphaseFidelities {
    double P_success;
    double F_swap;
    double F_CJ;
    double F_CJ_conditional;
    double single_photon_storage_retrieval_eff;
};

struct CphaseGateFidelityParameters
{
    int NAtoms;
    double g1d;
    double sigma;
    double delta;
    double Deltac;
    double kd_ensemble;
    double OmegaScattering;
    double OmegaStorageRetrieval;
    double kL1;
    double kL2;
    double scatteredPulseFrequencyWidth;
    double tNoInteractionAbs;
    double tNoInteractionArg;
    double t_storage;
    double t_retrieval;
    double t_to_pass_ensemble;
    std::vector<unsigned long int> randomSeeds;
    int periodLength;
    int impurityShift;
    std::string dumpDataDir;
    // Default initialize Deltac and sigma to zero,
    // since the optimization code expects this values
    // to signify that it needs to come up with a better
    // guess. Otherwise that code will asume that the values
    // are the initial guess values for the optimization.
    CphaseGateFidelityParameters() : Deltac(0), sigma(0) {}
};

// This struct is used for the optimization
// of repeated calls to
// 'calculate_store_eit_multiply_reflection_coefficient_retrieve'
// The matrix of the retrieval kernel, along with nodes
// and weights are cached in an array of these structs.
// It is only used for the adiabatic EIT retrieval kernels.
struct QuadratureK_r
{
    Eigen::MatrixXcd K_r;
    std::vector<Eigen::MatrixXcd> K_r_for_each_thread;
    ThreadedEigenMatrix K_r_threaded;
    Eigen::ArrayXd tVals;
    Eigen::ArrayXd tWeights;

    // This element is for convenience of
    // the code that uses the cache. The idea is
    // that *either* K_r or K_r_threaded is used
    // depending on which level we want to parallelize
    // the computations on. Then it is inconvenient to
    // decide each time, whether it is the number of
    // columns of K_r or K_r_threaded that need to be
    // checked. (And we want to check NAtoms to be able
    // to determine, whether the cache can be used or
    // has been computed for a different NAtoms and hence
    // needs to be discarded.)
    int NAtoms;
};

void set_stored_spin_wave_set_tolerances(double &dtInitial,
                                         double &normThreshold,
                                         double &fidThreshold,
                                         bool onlyAnalyticalCalculation);

Eigen::VectorXcd calculate_stored_spinwave_eit_field_elimination(
        LambdaHamiltonian1Excitation &H_EIT,
        EvolutionMethod method,
        double t_storage,
        double dtInitial,
        double normThreshold,
        int flags);

std::vector<std::complex<double>>
calculate_stored_spinwave_eit_dispersion_relation_plus(
        HamiltonianParams eitParams,
        bool forceRegularPositions);

std::vector<std::complex<double>>
calculate_stored_spinwave_eit_dispersion_relation_minus(
        HamiltonianParams eitParams,
        bool forceRegularPositions);

std::vector<std::complex<double>>
calculate_stored_spinwave_eit_kernel_plus(
        HamiltonianParams eitParams,
        double t_storage,
        double storageTolAbs,
        double storageTolRel,
        bool forceRegularPositions);

std::vector<std::complex<double>>
calculate_stored_spinwave_eit_kernel_minus(
        HamiltonianParams eitParams,
        double t_storage,
        double storageTolAbs,
        double storageTolRel,
        bool forceRegularPositions);

void calculate_store_eit_multiply_reflection_coefficient_retrieve(
        SpinwaveAndFieldVector &sol,
        CphaseGateFidelityParameters *cphaseParameters,
        int cphaseGateFlags, bool onlyAnalyticalCalculation,
        EnsembleScattering *ensemble_scattering,
        std::vector<QuadratureK_r> &quadK_r);

void calculate_cphase_fidelities_for_tNoInteraction(
        const SpinwaveAndFieldVector &sol,
        CphaseFidelities *fid,
        const std::vector<std::complex<double>> &noImpR,
        std::complex<double> tNoInteraction,
        int cphaseGateFlags,
        bool onlyAnalyticalCalculation);

void calculate_cphase_fidelities_spinwave(
        CphaseFidelities *fid,
        const SpinwaveAndFieldVector &sol,
        const std::vector<std::complex<double>> &noImpR,
        std::complex<double> tNoInteraction);

void calculate_cphase_fidelities_E_field(
        CphaseFidelities *fid,
        const SpinwaveAndFieldVector &sol,
        const std::vector<std::complex<double>> &noImpR,
        std::complex<double> tNoInteraction);

void calculate_cphase_fidelities_E_field_with_weights(
        CphaseFidelities *fid,
        const SpinwaveAndFieldVector &sol,
        const std::vector<std::complex<double>> &noImpR,
        std::complex<double> tNoInteraction);

double find_optimal_tNoInteraction(double *rho, double *theta,
                                   const SpinwaveAndFieldVector &sol,
                                   const std::vector<std::complex<double> > &noImpR,
                                   int cphaseGateFlags,
                                   bool onlyAnalyticalCalculation,
                                   bool globalOptimization = false,
                                   double xtol_abs = 1e-12);

struct CphaseDiagnosticData
{
    double E_without_scattering_norm;
    double E_with_scattering_norm;
    double spinwave_after_storage_norm;
    double spinwave_after_scattering_norm;
    std::complex<double> E_without_scattering_avg_overlap;
    std::complex<double> impurityRfromE;
    std::complex<double> impurityRfromSpinwave;
    std::complex<double> noImpurityR;
};

void calculate_cphase_fidelities_numerical_for_optimal_Deltac_sigma(
        CphaseFidelities *fid, CphaseFidelities *fid_tNoInteraction_one,
        SpinwaveAndFieldVector &sol,
        std::vector<CphaseDiagnosticData> &diagData,
        CphaseGateFidelityParameters *cphaseParameters, int cphaseGateFlags,
        EnsembleScattering *ensemble_scattering);

#endif // CPHASE_GATE_H
