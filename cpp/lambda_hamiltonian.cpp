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

#include "lambda_hamiltonian.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <omp.h>

#include "Eigen/Dense"
#include "unsupported/Eigen/MatrixFunctions"

#include "gaussian_modes.h"
#include "gaussian_electric_field_modes.h"
#include "threaded_eigen_matrix.h"

#include "urandom.h"

using Eigen::MatrixXcd;
using Eigen::VectorXcd;

namespace {
class HamiltonianStoreOperation
{
    Eigen::MatrixXcd &m_H;
public:
    inline explicit HamiltonianStoreOperation(Eigen::MatrixXcd &H) :
        m_H(H)
    {}
    inline void operator()(int i, int j, std::complex<double> val)
    {
        m_H(i,j) += val;
    }
    inline void operator()(int i, int j, const Eigen::VectorXcd &vals) const
    {
        const int vals_size = vals.size();
        for (int k = 0; k < vals_size; ++k) {
            m_H(i,j+k) += vals(k);
        }
    }
};

class HamiltonianMultiplyOnVectorOperation
{
    Eigen::VectorXcd &m_inputVector;
    Eigen::VectorXcd &m_outputVector;
public:
    inline HamiltonianMultiplyOnVectorOperation(Eigen::VectorXcd &outputVector,
                                                Eigen::VectorXcd &inputVector) :
        m_inputVector(inputVector),
        m_outputVector(outputVector)
    {}
    inline void operator()(int i, int j, std::complex<double> val) const
    {
        m_outputVector[i] += val*m_inputVector[j];
    }
    inline void operator()(int i, int j, const Eigen::VectorXcd &vals) const
    {
        m_outputVector[i] += m_inputVector.segment(j,vals.size()).cwiseProduct(vals).sum();
    }
};
} // unnamed namespace

class LambdaHamiltonian1ExcitationMatrix
{
public:
    HamiltonianParams m_params;
    int m_NBase;
    Eigen::MatrixXcd m_expMatrix;
    std::vector<Eigen::MatrixXcd> m_expMatrices;
    ThreadedEigenMatrix m_expMatrixThreaded;
    std::vector<int> m_threadNumaNodes;
    Eigen::VectorXcd m_expTable;
    Eigen::VectorXcd m_OmegaTable;
    Eigen::VectorXcd m_E_R_factors;
    Eigen::VectorXcd m_E_L_factors;
    Eigen::MatrixXcd m_psi_intermediate_snapshots;
    Eigen::VectorXcd m_E_R_input_data;
    Eigen::VectorXcd m_E_L_input_data;
    Eigen::VectorXcd m_E_R;
    Eigen::VectorXcd m_E_L;
    Eigen::VectorXcd m_E_R_free;
    Eigen::VectorXcd m_E_L_free;
    Eigen::VectorXd m_bStatesNorm;
    Eigen::VectorXd m_cStatesNorm;
    Eigen::VectorXd m_cStatesMean;
    Eigen::VectorXd m_tVals;
    Eigen::VectorXcd m_E_R_Extra;
    Eigen::VectorXcd m_E_L_Extra;
    Eigen::VectorXcd m_E_R_free_Extra;
    Eigen::VectorXcd m_E_L_free_Extra;
    Eigen::VectorXd m_tVals_Extra;
    int m_impurityAtomIndex;
    std::complex<double> m_inputFieldFactor;
    Eigen::MatrixXcd m_H;

    LambdaHamiltonian1ExcitationMatrix() :
        m_params(),
        m_impurityAtomIndex(0),
        m_H()
    {}
    void setParams(const HamiltonianParams &params);
    VectorXcd evolveExp(const VectorXcd &initial, double t);
    VectorXcd evolveRK(const VectorXcd &initial, double t, double dt, int flags);
    bool evolveRKOnce(VectorXcd &final, const VectorXcd &initial, double t,
                      unsigned N_t, int flags);
    bool evolveRKOnceRegular(VectorXcd &final, const VectorXcd &initial, double t,
                             unsigned N_t, int flags);
    bool evolveRKOnceRandomOptimized(VectorXcd &final, const VectorXcd &initial, double t,
                                     unsigned N_t, int flags);
    Eigen::VectorXcd E_R() const { return m_E_R; }
    Eigen::VectorXcd E_L() const { return m_E_L; }
    Eigen::VectorXcd E_R_free() const { return m_E_R_free; }
    Eigen::VectorXcd E_L_free() const { return m_E_L_free; }
    Eigen::VectorXd bStatesNorm() const { return m_bStatesNorm; }
    Eigen::VectorXd cStatesNorm() const { return m_cStatesNorm; }
    Eigen::VectorXd cStatesMean() const { return m_cStatesMean; }
    Eigen::VectorXd tVals() const { return m_tVals; }
    Eigen::MatrixXcd intermediateSolutions() const
    {
        return m_psi_intermediate_snapshots;
    }
    void electricFieldRL(std::complex<double> &E_R,
                         std::complex<double> &E_L,
                         std::complex<double> &E_R_free,
                         std::complex<double> &E_L_free,
                         const Eigen::VectorXcd &state,
                         double t);
    void electricFieldRLFixedInput(std::complex<double> &E_R,
                                   std::complex<double> &E_L,
                                   std::complex<double> &E_R_free,
                                   std::complex<double> &E_L_free,
                                   const Eigen::VectorXcd &state,
                                   int n);
    int NBase() const { return m_NBase; }
    template <class Function>
    void operateWithHamiltonian(Function f) const
    {
        const int NAtoms = m_params.NAtoms;

        const std::complex<double> I(0,1);

        const double Delta = m_params.Deltac+m_params.delta;
        // Gamma'/(Gamma_1D+Gamma') = 1 - Gamma_1D/(Gamma_1D+Gamma')
        // <b_i a|H|b_j a>
        if (m_params.randomAtomPositions) {
            #pragma omp parallel for
            for (int i = 0; i < NAtoms; ++i) {
                for (int j = 0; j < NAtoms; ++j) {
                    f(i, j, m_expMatrix(i, j));
                    //ret(i) += m_expMatrix(i, j)*vec(j);
                    //m_H(i, j) = m_expMatrix(i, j);
                }
                // The diagonal element has been absorbed into m_expMatrix

                //<b_i a|H|c_i a>
                f(i, i + NAtoms, -m_OmegaTable[i]);
                //ret(i) -= m_OmegaTable[i]*vec(i + NAtoms);
                //m_H(i + NAtoms, i) = -std::conj(m_OmegaTable[i]);

                // <c_i a|H|b_i a>
                f(i + NAtoms, i, -std::conj(m_OmegaTable[i]));
                //ret(i + NAtoms) -= std::conj(m_OmegaTable[i])*vec(i);
                //m_H(i, i + NAtoms) = -m_OmegaTable[i];

                // <c_i a|H|c_i a>
                f(i + NAtoms, i + NAtoms, -m_params.delta);
                //ret(i) -= m_params.delta*vec(i);
                //m_H(i, i) = -m_params.delta;
            }
        } else {
            if (m_params.deltaValues.size() != m_params.NAtoms) {
                const std::complex<double> DeltaGamma
                    = -(Delta + I*0.5*(1-m_params.g1d));
                #pragma omp parallel for
                for (int i = 0; i < NAtoms; ++i) {
#if 0
                    // This is the reference code, but
                    // std::abs() call seems to be expensive
                    // so we split the loop into two parts
                    // below such that we can avoid it.
                    for (int j = 0; j < NAtoms; ++j) {
                        const int indexDiff = std::abs(i-j);
                        f(i, j, m_expTable[indexDiff]);
                        //ret(i) += m_expTable[indexDiff]*vec(j);
                        //m_H(i, j) = m_expTable[indexDiff];
                    }
#endif // 0
#if 0
                    // This is the part of the loop for j < i
                    // Below, it is expressed as a faster version
                    // using subvectors of m_expTable. This allows
                    // Eigen to use the best possible vectorization
                    // option automatically.
                    int j = 0;
                    for (; j < i; ++j) {
                        const int indexDiff = i-j;
                        f(i, j, m_expTable[indexDiff]);
                        //ret(i) += m_expTable[indexDiff]*vec(j);
                        //m_H(i, j) = m_expTable[indexDiff];
                    }
#endif // 0
                    if (i > 0) {
                        f(i, 0, m_expTable.segment(1,i).reverse());
                    }
#if 0
                    // This is the part of the loop for j >= i
                    // Below, it is expressed as a faster version
                    // using subvectors of m_expTable.
                    for (; j < NAtoms; ++j) {
                        const int indexDiff = j-i;
                        f(i, j, m_expTable[indexDiff]);
                        //ret(i) += m_expTable[indexDiff]*vec(j);
                        //m_H(i, j) = m_expTable[indexDiff];
                    }
#endif // 0
                    f(i, i, m_expTable.head(NAtoms-i));

                    f(i, i, DeltaGamma);
                    //ret(i) += -(m_params.Delta + I*0.5*(1-m_params.g1d))*vec(i);
                    //m_H(i, i) += -(params.Delta + I*0.5*(1-params.g1d));

                    //<b_i a|H|c_i a>
                    f(i, i + NAtoms, -m_OmegaTable[i]);
                    //ret(i) -= m_OmegaTable[i]*vec(i + NAtoms);
                    //m_H(i + NAtoms, i) = -std::conj(m_OmegaTable[i]);

                    // <c_i a|H|b_i a>
                    f(i + NAtoms, i, -std::conj(m_OmegaTable[i]));
                    //ret(i + NAtoms) -= std::conj(m_OmegaTable[i])*vec(i);
                    //m_H(i, i + NAtoms) = -m_OmegaTable[i];

                    // <c_i a|H|c_i a>
                    f(i + NAtoms, i + NAtoms, -m_params.delta);
                    //ret(i) -= m_params.delta*vec(i);
                    //m_H(i, i) = -m_params.delta;
                }
            } else {
                #pragma omp parallel for
                for (int i = 0; i < NAtoms; ++i) {
                    const double Delta_i = m_params.Deltac+m_params.deltaValues[i];
                    const std::complex<double> DeltaGamma
                        = -(Delta_i + I*0.5*(1-m_params.g1d));
#if 0
                    // This is the reference code, but
                    // std::abs() call seems to be expensive
                    // so we split the loop into two parts
                    // below such that we can avoid it.
                    for (int j = 0; j < NAtoms; ++j) {
                        const int indexDiff = std::abs(i-j);
                        f(i, j, m_expTable[indexDiff]);
                        //ret(i) += m_expTable[indexDiff]*vec(j);
                        //m_H(i, j) = m_expTable[indexDiff];
                    }
#endif // 0
#if 0
                    // This is the part of the loop for j < i
                    // Below, it is expressed as a faster version
                    // using subvectors of m_expTable. This allows
                    // Eigen to use the best possible vectorization
                    // option automatically.
                    int j = 0;
                    for (; j < i; ++j) {
                        const int indexDiff = i-j;
                        f(i, j, m_expTable[indexDiff]);
                        //ret(i) += m_expTable[indexDiff]*vec(j);
                        //m_H(i, j) = m_expTable[indexDiff];
                    }
#endif // 0
                    if (i > 0) {
                        f(i, 0, m_expTable.segment(1,i).reverse());
                    }
#if 0
                    // This is the part of the loop for j >= i
                    // Below, it is expressed as a faster version
                    // using subvectors of m_expTable.
                    for (; j < NAtoms; ++j) {
                        const int indexDiff = j-i;
                        f(i, j, m_expTable[indexDiff]);
                        //ret(i) += m_expTable[indexDiff]*vec(j);
                        //m_H(i, j) = m_expTable[indexDiff];
                    }
#endif // 0
                    f(i, i, m_expTable.head(NAtoms-i));

                    f(i, i, DeltaGamma);
                    //ret(i) += -(m_params.Delta + I*0.5*(1-m_params.g1d))*vec(i);
                    //m_H(i, i) += -(params.Delta + I*0.5*(1-params.g1d));

                    //<b_i a|H|c_i a>
                    f(i, i + NAtoms, -m_OmegaTable[i]);
                    //ret(i) -= m_OmegaTable[i]*vec(i + NAtoms);
                    //m_H(i + NAtoms, i) = -std::conj(m_OmegaTable[i]);

                    // <c_i a|H|b_i a>
                    f(i + NAtoms, i, -std::conj(m_OmegaTable[i]));
                    //ret(i + NAtoms) -= std::conj(m_OmegaTable[i])*vec(i);
                    //m_H(i, i + NAtoms) = -m_OmegaTable[i];

                    // <c_i a|H|c_i a>
                    f(i + NAtoms, i + NAtoms, -m_params.delta);
                    //ret(i) -= m_params.delta*vec(i);
                    //m_H(i, i) = -m_params.delta;
                }
            }
        }

        // The "impurity" mode below doesn't really make sense if we
        // only effectively have two-level atoms instead of Lambda-type
        // atoms, as it would have replaced one Lambda-type atom by a
        // two-level atom.
        if (!m_params.disable_classical_drive) {
            if (m_params.putImpurity) {
                // Make the detuning on the impurity atom Delta=0
                // by subtracting the detuning added in the above loop
                f(m_impurityAtomIndex, m_impurityAtomIndex, Delta);
                //ret(m_impurityAtomIndex) += m_params.Delta*vec(m_impurityAtomIndex);
                //m_H(impurityAtomIndex, impurityAtomIndex) += params.Delta;
            }
        }
    }
    inline std::complex<double> inputE_R(double z, double t)
    {
        if (m_params.input_electric_field_R_factor == 0.0) {
            return 0.0;
        }
        const std::complex<double> I(0,1);
        std::complex<double> factor = m_params.input_electric_field_R_factor;
        if (m_params.input_electric_field_L_factor != 0.0) {
            factor *= M_SQRT1_2;
        }
        const double k_in = m_params.kd_ensemble*M_PI*m_params.NAtoms;
        return factor
               *gaussian_input_electric_field_mode(t-z/m_params.cOverLGamma,
                                                   m_params.inputE_mean,
                                                   m_params.inputE_width)
               *std::exp(I*k_in*z);
    }
    inline std::complex<double> inputE_L(double z, double t)
    {
        if (m_params.input_electric_field_L_factor == 0.0) {
            return 0.0;
        }
        std::complex<double> factor = m_params.input_electric_field_L_factor;
        if (m_params.input_electric_field_R_factor != 0.0) {
            factor *= M_SQRT1_2;
        }
        const std::complex<double> I(0,1);
        const double k_in = m_params.kd_ensemble*M_PI*m_params.NAtoms;
        return factor
               *gaussian_input_electric_field_mode(t-(1-z)/m_params.cOverLGamma,
                                                   m_params.inputE_mean,
                                                   m_params.inputE_width)
               *std::exp(-I*k_in*z);
    }

    inline std::complex<double> inputE(double z, double t)
    {
        return m_inputFieldFactor*(inputE_R(z, t)+inputE_L(z, t));
    }

    void updateClassicalDrive(double a, double currentT, double maxT);
    void reallocateRKdata(int N_t, bool compute_electric_field,
                          bool compute_spinwave_norm);
};

void LambdaHamiltonian1ExcitationMatrix::setParams(const HamiltonianParams &params)
{
    assert(params.NAtoms > 0 && "Number of atoms must be at least one!");

    m_params = params;
    const int NAtoms = m_params.NAtoms;

    // Compute dimension of the Hilbert space

    // Number of single excitation states is equal to the number of atoms

    m_NBase = 2*NAtoms;

    const double gridSpacing = params.gridSpacing;
    const double k_in = m_params.kd_ensemble*M_PI*m_params.NAtoms;
    const std::complex<double> I(0,1);

    if (m_params.atom_positions.empty()) {
        m_params.atom_positions = std::vector<double>(NAtoms);
        if (!m_params.randomAtomPositions) {
            fill_in_regular_atomic_positions(m_params.atom_positions.data(),
                                             NAtoms, gridSpacing);
        } else {
            std::random_device randomDevice;
            std::mt19937 generator(randomDevice());
            generate_random_atom_positions(m_params.atom_positions.data(), generator, NAtoms);
        }
    }

    m_inputFieldFactor = I*std::sqrt(m_params.g1d/2);
    m_E_R_factors = Eigen::VectorXcd::Zero(NAtoms);
    m_E_L_factors = Eigen::VectorXcd::Zero(NAtoms);
    const double L = 1;
    const std::complex<double> g1dFactor = I*std::sqrt(m_params.g1d/(2));
    for (int i = 0; i < NAtoms; ++i) {
        const double zi = m_params.atom_positions[i];
        // E_R(z=L)
        m_E_R_factors[i] = g1dFactor*std::exp(-I*k_in*zi);
        // E_L(z=0)
        m_E_L_factors[i] = g1dFactor*std::exp(I*k_in*zi);
    }

    if (m_params.use_input_electric_field_data
            && m_params.E_R_input_data.size() == 0
            && m_params.E_R_input_data.size() == 0) {
        std::cout << "No input E-field data given. Using Gaussian mode input."
                  << std::endl;
        m_params.use_input_electric_field_data = false;
    }

    if (m_params.use_input_electric_field_data) {
        const int input_E_R_size = m_params.E_R_input_data.size();
        const int input_E_L_size = m_params.E_L_input_data.size();
        if (input_E_R_size != 0 && input_E_L_size == 0) {
            m_E_R_input_data = Eigen::VectorXcd::Zero(input_E_R_size);
            m_E_L_input_data = Eigen::VectorXcd::Zero(input_E_R_size);
            for (int i = 0; i < input_E_R_size; ++i) {
                m_E_R_input_data[i] = m_params.E_R_input_data[i];
            }
        } else if (input_E_R_size == 0 && input_E_L_size != 0) {
            m_E_R_input_data = Eigen::VectorXcd::Zero(input_E_L_size);
            m_E_L_input_data = Eigen::VectorXcd::Zero(input_E_L_size);
            for (int i = 0; i < input_E_R_size; ++i) {
                m_E_L_input_data[i] = m_params.E_L_input_data[i];
            }
        } else if (input_E_R_size != 0 && input_E_R_size == input_E_L_size) {
            m_E_R_input_data = Eigen::VectorXcd::Zero(input_E_R_size);
            m_E_L_input_data = Eigen::VectorXcd::Zero(input_E_R_size);
            for (int i = 0; i < input_E_R_size; ++i) {
                m_E_R_input_data[i] = m_params.E_R_input_data[i];
                m_E_L_input_data[i] = m_params.E_L_input_data[i];
            }
        } else {
            m_params.use_input_electric_field_data = false;
            std::cout << "Unsupported combination of the input E-field vectors:"
                      << '\n' << "E_R vector size = " << input_E_R_size
                      << ", E_L_vector_size = " << input_E_L_size << std::endl;
            assert(0 && "Unsupported combination of the input E-field vectors");
        }
    }

    // Here we still special case the random and the regularly spaced
    // atom positions, since the regularly spaced case needs fewer
    // stored elements.
    if (!params.randomAtomPositions) {
        m_expTable = Eigen::VectorXcd::Zero(NAtoms);
        for (int i = 0; i < NAtoms; ++i) {
            const double distance = position_from_index(i, gridSpacing);
            const double phase = k_in*distance;
            m_expTable[i] = -0.5*I*m_params.g1d*std::exp(I*phase);
        }
    } else {
        assert(!m_params.atom_positions.empty()
               && "Atom positions array should have been provided!");
        const std::complex<double> DeltaGamma
            = -(m_params.Deltac+m_params.delta + I*0.5*(1-m_params.g1d));
        m_expMatrix = MatrixXcd::Zero(NAtoms, NAtoms);
        for (int i = 0; i < NAtoms; ++i) {
            for (int j = 0; j < NAtoms; ++j) {
                const double distance = std::abs(m_params.atom_positions[i]
                                                 -m_params.atom_positions[j]);
                const double phase = k_in*distance;
                m_expMatrix(i,j) = -0.5*I*m_params.g1d*std::exp(I*phase);
            }
            m_expMatrix(i, i) += DeltaGamma;
        }
        // The splitted matrix below is used for the
        // random placement calculation with the more efficient
        // method.
        m_expMatrixThreaded = ThreadedEigenMatrix(m_expMatrix);
    }

    if (m_params.classicalDriveOppositeDetunings) {
        // First of all this option only works for the
        // counterpropagating setup.
        m_params.counterpropagating = true;

        // The classical drives detuned in different directions
        // result in the time dependent phases for the forward and
        // and backward components of the fields. Thus makes little
        // sense to not put the spatial phases in the standing wave.
        m_params.putPhasesInTheClassicalDrive = true;
    }

    if (m_OmegaTable.size() == 0) {
        m_OmegaTable = Eigen::VectorXcd(params.NAtoms);
    }

    // Here we use the m_params.atom_positions array both for
    // the random and the regularly spaced atom positions
    if (params.counterpropagating) {
        for (int i = 0; i < NAtoms; ++i) {
            const double distance = m_params.atom_positions[i];
            m_OmegaTable[i] = 2.0*m_params.Omega*std::cos(k_in*distance);
        }
    } else {
        if (m_params.putPhasesInTheClassicalDrive) {
            for (int i = 0; i < NAtoms; ++i) {
                const double distance = m_params.atom_positions[i];
                m_OmegaTable[i] = params.Omega*exp(I*k_in*distance);
            }
        } else {
            for (int i = 0; i < NAtoms; ++i) {
                m_OmegaTable[i] = params.Omega;
            }
        }
    }
    m_impurityAtomIndex = NAtoms / 2;
    if (params.putImpurity) {
        if (params.impurity_position < 0) {
            // Find an atom that was setup to have
            // a classical drive on by the rules above.
            // (Remember in the counterpropagating setup
            // only every second atom has the classical
            // drive turned on). I.e. skip the atom
            // that doesn't have the classical drive turned on.
            while (std::abs(m_OmegaTable[m_impurityAtomIndex]) < 1e-9) {
                ++m_impurityAtomIndex;
            }
        } else {
            // Use the impurity index that was given
            // to us as a parameter
            m_impurityAtomIndex = params.impurity_position;
        }
        m_OmegaTable[m_impurityAtomIndex] = 0.0;
    }
}

VectorXcd LambdaHamiltonian1ExcitationMatrix::evolveExp(const VectorXcd &initial, double t)
{
    // This method doesn't support time-dependent Hamiltonians,
    // i.e. where, say, Omega changes as a function of time
    // (adiabatically ramped up or down). Moreover it's difficult
    // (but should be possible) to include input electric field here.
    if (m_H.rows() != m_NBase || m_H.cols() == m_NBase) {
        // Fill in Hamiltonian matrix
        m_H = MatrixXcd::Zero(m_NBase, m_NBase);
        HamiltonianStoreOperation operation(m_H);
        operateWithHamiltonian(operation);
    }
    return (std::complex<double>(0,-1)*m_H*t).exp()*initial;
}

namespace {
inline double OmegaAdiabaticTurnOnFactor(double a, double tFactor)
{
    if (tFactor < a) {
        return (double(1)/a)*tFactor;
    } else {
        return 1;
    }
}
} // unnamed namespace

void
LambdaHamiltonian1ExcitationMatrix::updateClassicalDrive(
        double a,
        double currentT,
        double maxT)
{
    // We only need to do something here if:
    //
    // 1. We are in the counterpropagating EIT,
    //    and we adiabatically go to the regular EIT.
    // 2. We are in the regular EIT,
    //    and we adiabatically go to the counterpropagating EIT.
    const bool changeStrength = (m_params.counterpropagating && m_params.adiabaticallyGoToRegularEit)
        || (!m_params.counterpropagating && m_params.adiabaticallyGoToCounterpropagatingEit);
    if (!changeStrength && !m_params.classicalDriveOppositeDetunings) {
        return;
    }
    assert(m_OmegaTable.size() > 0 && "m_OmegaTable was not allocated!");
    double totalOmegaFactor = 1;
    double OmegaFactor = 1;
    const std::complex<double> I(0,1);
    if (changeStrength) {
        const double tFactor = currentT/maxT;
        totalOmegaFactor = OmegaAdiabaticTurnOnFactor(a, tFactor);
        const double b = 0.45;
        double OmegaFactor;
        /*
        if (tFactor < a) {
            OmegaFactor = 1;
        } else {
            //const double A = double(1)/(a-1);
            //const double B = -A;
            const double A = -double(1)/b;
            const double B = -(a+b)*A;
            OmegaFactor = A*tFactor+B;
        }
        */
        OmegaFactor = tFactor; // TESTING!
        //OmegaFactor = 1;
    }

    // Note that
    // |e^{ikz} + OmegaFactor*e^{-ikz}|^2
    // = 1 + OmegaFactor^2 + 2*OmegaFactor*cos(2*kz)
    const double OmegaFactor2 = OmegaFactor*OmegaFactor;

    // Here we use the m_params.atom_positions array both for
    // the random and the regularly spaced atom positions
    if (m_params.putPhasesInTheClassicalDrive) {
        std::complex<double> timePhase = 1.0;
        if (m_params.classicalDriveOppositeDetunings) {
            //timePhase = std::exp(-I*m_params.Deltac*currentT);
        }
        const double k_c = m_params.kd_ensemble*M_PI*m_params.NAtoms;
        for (int i = 0; i < m_params.NAtoms; ++i) {
            const double distance = m_params.atom_positions[i];
            m_OmegaTable[i] = totalOmegaFactor*m_params.Omega
                              *(std::exp(I*k_c*distance)*timePhase
                                +OmegaFactor*std::exp(-I*k_c*distance)*std::conj(timePhase));
        }
    } else {
        for (int i = 0; i < m_params.NAtoms; ++i) {
            const double distance = m_params.atom_positions[i];
            const double phase = m_params.kd_ensemble*M_PI*m_params.NAtoms*distance;
            //m_OmegaTable[i] = totalOmegaFactor*m_params.Omega*(2*(1-OmegaFactor)+2*std::cos(phase)*OmegaFactor);
            m_OmegaTable[i] = totalOmegaFactor*m_params.Omega*std::sqrt(1+OmegaFactor2+2*OmegaFactor*std::cos(phase));
        }
    }
    if (m_params.putImpurity) {
        m_OmegaTable[m_impurityAtomIndex] = 0.0;
    }
}


inline void
LambdaHamiltonian1ExcitationMatrix::electricFieldRL(std::complex<double> &E_R,
                                                    std::complex<double> &E_L,
                                                    std::complex<double> &E_R_free,
                                                    std::complex<double> &E_L_free,
                                                    const Eigen::VectorXcd &state,
                                                    double t)
{
    // We are finding the right-going field at the position
    // z=L. Thus in the general formula
    // E_R = \sum_j \theta_H(z-z_j) S_ab^j(t) e^{ik(z-z_j)
    // all the Heaviside theta functions \theta_H are equal
    // to 1.
    // Similarly, in
    // E_L = \sum_j \theta_H(z_j-z) S_ab^j(t) e^{ik(z_j-z)}
    // all the theta functions become 1 too because we evaluate
    // the left-going field at z=0
    E_R = std::complex<double>(0,0);
    E_L = std::complex<double>(0,0);
    E_R_free = std::complex<double>(0,0);
    E_L_free = std::complex<double>(0,0);

    E_R += state.head(m_params.NAtoms).cwiseProduct(m_E_R_factors).sum();
    E_L += state.head(m_params.NAtoms).cwiseProduct(m_E_L_factors).sum();

    // The z argument is rescaled, so that it is
    // actually z/L=1 and z/L=0 respectively.
    E_R += inputE_R(1, t);
    E_R_free += inputE_R(1, t);
    E_L += inputE_L(0, t);
    E_L_free += inputE_L(0, t);
}

inline void
LambdaHamiltonian1ExcitationMatrix::electricFieldRLFixedInput(
        std::complex<double> &E_R,
        std::complex<double> &E_L,
        std::complex<double> &E_R_free,
        std::complex<double> &E_L_free,
        const Eigen::VectorXcd &state,
        int n)
{
    // We are finding the right-going field at the position
    // z=L. Thus in the general formula
    // E_R = \sum_j \theta_H(z-z_j) S_ab^j(t) e^{ik(z-z_j)
    // all the Heaviside theta functions \theta_H are equal
    // to 1.
    // Similarly, in
    // E_L = \sum_j \theta_H(z_j-z) S_ab^j(t) e^{ik(z_j-z)}
    // all the theta functions become 1 too because we evaluate
    // the left-going field at z=0
    E_R = std::complex<double>(0,0);
    E_L = std::complex<double>(0,0);
    E_R_free = std::complex<double>(0,0);
    E_L_free = std::complex<double>(0,0);
    E_R += state.head(m_params.NAtoms).cwiseProduct(m_E_R_factors).sum();
    E_L += state.head(m_params.NAtoms).cwiseProduct(m_E_L_factors).sum();

    E_R += m_E_R_input_data[n];
    E_R_free += m_E_R_input_data[n];
    E_L += m_E_L_input_data[n];
    E_L_free += m_E_L_input_data[n];
}

namespace {
inline int aligned_number(int N, int k) {
    int quotient = N / k;
    const int remainder = N % k;
    if (remainder == 0) {
        return N;
    }
    return (quotient+1)*k;
}
} // unnamed namespace

VectorXcd
LambdaHamiltonian1ExcitationMatrix::evolveRK(const VectorXcd &initial,
                                              double t, double dt, int flags)
{
    int N_t;
    N_t = int(t/dt);

    bool store_intermediate_spinwave = false;

    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_STORE_INTERMEDIATE_SPINWAVE_STATE)
    {
        store_intermediate_spinwave = true;
        // There shouldn't be more snapshots than the time steps
        if (m_params.intermediate_spinwave_snapshots > N_t) {
            m_params.intermediate_spinwave_snapshots = N_t;
        }

        // Align N_t with a multiple of the snapshot number
        N_t = aligned_number(N_t, m_params.intermediate_spinwave_snapshots);
    }

    const double epsilon = 1e-4;
    VectorXcd ret(initial.size());
    bool success;
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_RESTART_IF_BOGUS_RESULTS) {
        for (int i = 0; i < 10; ++i) {
            std::cout << " i = " << i << ", N_t = " << N_t << std::endl;
            std::cout << " dt = " << dt << std::endl;
            success = evolveRKOnce(ret, initial, t, N_t, flags);
            double retNorm = ret.norm();
            if (success && retNorm <= 1+epsilon) {
                if (retNorm > 1) {
                    std::cout << "  Normalizing the returned state. (norm diff = "
                              << retNorm-1 << ")" << std::endl;
                    ret /= retNorm;
                }
                return ret;
            }
            if (retNorm > 1+epsilon) {
                std::cout << "  RK final norm diff = " << retNorm-1
                          << " (for g1d = " << m_params.g1d << ")" << std::endl;
            }
            N_t *= 2;

            if (store_intermediate_spinwave) {
                // Align N_t with a multiple of the snapshot number
                N_t = aligned_number(N_t, m_params.intermediate_spinwave_snapshots);
            }
        }
        std::cout << "   Runge-Kutta process failed converging!" << std::endl;
    } else {
        evolveRKOnce(ret, initial, t, N_t, flags);
    }
    return ret;
}

void
LambdaHamiltonian1ExcitationMatrix::reallocateRKdata(int N_t,
        bool compute_electric_field, bool compute_spinwave_norm)
{
    // All the arrays here are always allocated
    // to have the same size. That's why we only
    // check the size of one of them.

    int vec_size = m_E_R.size();
    if (compute_electric_field) {
        if (vec_size > 0) {
            m_E_R.conservativeResize(N_t);
            m_E_L.conservativeResize(N_t);
            m_E_R_free.conservativeResize(N_t);
            m_E_L_free.conservativeResize(N_t);
            m_tVals.conservativeResize(N_t);
        } else {
            m_E_R = VectorXcd::Zero(N_t);
            m_E_L = VectorXcd::Zero(N_t);
            m_E_R_free = VectorXcd::Zero(N_t);
            m_E_L_free = VectorXcd::Zero(N_t);
            m_tVals = Eigen::VectorXd::Zero(N_t);
        }
    } else {
        m_E_R = VectorXcd();
        m_E_L = VectorXcd();
        m_E_R_free = VectorXcd();
        m_E_L_free = VectorXcd();
    }
    if (compute_spinwave_norm) {
        if (vec_size > 0) {
            m_cStatesNorm.conservativeResize(N_t);
            m_cStatesMean.conservativeResize(N_t);
        } else {
            m_cStatesNorm = Eigen::VectorXd::Zero(N_t);
            m_cStatesMean = Eigen::VectorXd::Zero(N_t);
        }
    } else {
        m_cStatesNorm = Eigen::VectorXd();
        m_cStatesMean = Eigen::VectorXd();
    }
    // TODO: Handle bStatesNorm
}

bool
LambdaHamiltonian1ExcitationMatrix::evolveRKOnce(VectorXcd &final, const VectorXcd &initial,
                                                  double t, unsigned N_t, int flags)
{
    if (m_params.randomAtomPositions) {
        return evolveRKOnceRandomOptimized(final, initial, t, N_t, flags);
    }
    return evolveRKOnceRegular(final, initial, t, N_t, flags);
}

bool
LambdaHamiltonian1ExcitationMatrix::evolveRKOnceRegular(VectorXcd &final, const VectorXcd &initial,
                                                         double t, unsigned N_t, int flags)
{
    double dt = t/N_t;
    bool compute_electric_field = false;
    bool compute_spinwave_norm = false;
    bool compute_excited_norm = false;
    bool restart_if_bogus_results = false;
    bool store_intermediate_spinwave = false;
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_ELECTRIC_FIELD) {
        compute_electric_field = true;
    }
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_SPINWAVE_NORM) {
        compute_spinwave_norm = true;
    }
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_EXCITED_NORM) {
        compute_excited_norm = true;
    }
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_RESTART_IF_BOGUS_RESULTS) {
        restart_if_bogus_results = true;
    }
    const int vec_size = initial.size();
    assert(vec_size == 2*m_params.NAtoms
           && "State vector does not have the length 2*NAtoms!");
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_STORE_INTERMEDIATE_SPINWAVE_STATE)
    {
        store_intermediate_spinwave = true;
        m_psi_intermediate_snapshots
                = Eigen::MatrixXcd::Zero(
                    vec_size,
                    m_params.intermediate_spinwave_snapshots);
    }
    if (compute_electric_field) {
        m_E_R = VectorXcd::Zero(N_t);
        m_E_L = VectorXcd::Zero(N_t);
        m_E_R_free = VectorXcd::Zero(N_t);
        m_E_L_free = VectorXcd::Zero(N_t);
    } else {
        m_E_R = VectorXcd();
        m_E_L = VectorXcd();
        m_E_R_free = VectorXcd();
        m_E_L_free = VectorXcd();
    }
    if (compute_spinwave_norm) {
        m_cStatesNorm = Eigen::VectorXd::Zero(N_t);
        m_cStatesMean = Eigen::VectorXd::Zero(N_t);
    } else {
        m_cStatesNorm = Eigen::VectorXd();
        m_cStatesMean = Eigen::VectorXd();
    }
    if (compute_excited_norm) {
        m_bStatesNorm = Eigen::VectorXd::Zero(N_t);
    } else {
        m_bStatesNorm = Eigen::VectorXd();
    }
    if (m_OmegaTable.size() == 0) {
        m_OmegaTable = Eigen::VectorXcd(m_params.NAtoms);
    }
    const std::complex<double> I(0,1);
    const double a = 0.05;

    int outer_loop_iterations = 1;
    int inner_loop_iterations = N_t;
    if (store_intermediate_spinwave) {
        outer_loop_iterations = m_params.intermediate_spinwave_snapshots;
        // The assumption is that the caller of this function
        // has arranged for the number of the snapshots to
        // be a divisor of the total number of iterations
        inner_loop_iterations = N_t / m_params.intermediate_spinwave_snapshots;
    }
    Eigen::VectorXcd psi = initial;

    Eigen::VectorXcd k1(vec_size);
    Eigen::VectorXcd k2(vec_size);
    Eigen::VectorXcd k3(vec_size);
    Eigen::VectorXcd k4(vec_size);
    Eigen::VectorXcd temp(vec_size);

    HamiltonianMultiplyOnVectorOperation operation1(k1, psi);
    HamiltonianMultiplyOnVectorOperation operation2(k2, temp);
    HamiltonianMultiplyOnVectorOperation operation3(k3, temp);
    HamiltonianMultiplyOnVectorOperation operation4(k4, temp);
    for (int n = 0; n < outer_loop_iterations; ++n) {
        int i_min = 0;
        int i_max = N_t;
        if (store_intermediate_spinwave) {
            i_min = n*inner_loop_iterations;
            i_max = i_min + inner_loop_iterations;
        }
        for (int i = i_min; i < i_max; ++i) {
            //k1=-1i*H*psi0*Dt;
            //k2=-1i*H*(psi0+k1*0.5)*Dt;
            //k3=-1i*H*(psi0+k2*0.5)*Dt;
            //k4=-1i*H*(psi0+k3)*Dt;
            //psi0 = psi0 + (k1/6) + (k2/3) + (k3/3) + (k4/6);

            const double currentT1 = i*dt;
            const double currentT2 = currentT1+dt/2;
            const double currentT3 = currentT2;
            const double currentT4 = currentT1+dt;

            k1.setZero();
            k2.setZero();
            k3.setZero();
            k4.setZero();

            updateClassicalDrive(a, currentT1, t);
            operateWithHamiltonian(operation1);

            updateClassicalDrive(a, currentT2, t);
            for (int j = 0; j < m_params.NAtoms; ++j) {
                const double zj = m_params.atom_positions[j];
                k1[j] *= -I*dt;
                k1[j] += inputE(zj, currentT1)*dt;
                temp[j] = psi[j] + 0.5*k1[j];
            }
            for (int j = m_params.NAtoms; j < vec_size; ++j) {
                k1[j] *= -I*dt;
                temp[j] = psi[j] + 0.5*k1[j];
            }
            operateWithHamiltonian(operation2);

            // The line below is commented out
            // because currentT3 == currentT2

            //updateClassicalDrive(a, currentT3, t);
            for (int j = 0; j < m_params.NAtoms; ++j) {
                const double zj = m_params.atom_positions[j];
                k2[j] *= -I*dt;
                k2[j] += inputE(zj, currentT2)*dt;
                temp[j] = psi[j] + 0.5*k2[j];
            }
            for (int j = m_params.NAtoms; j < vec_size; ++j) {
                k2[j] *= -I*dt;
                temp[j] = psi[j] + 0.5*k2[j];
            }
            operateWithHamiltonian(operation3);

            updateClassicalDrive(a, currentT4, t);
            for (int j = 0; j < m_params.NAtoms; ++j) {
                const double zj = m_params.atom_positions[j];
                k3[j] *= -I*dt;
                k3[j] += inputE(zj, currentT3)*dt;
                temp[j] = psi[j] + k3[j];
            }
            for (int j = m_params.NAtoms; j < vec_size; ++j) {
                k3[j] *= -I*dt;
                temp[j] = psi[j] + k3[j];
            }
            operateWithHamiltonian(operation4);

            for (int j = 0; j < m_params.NAtoms; ++j) {
                const double zj = m_params.atom_positions[j];
                k4[j] *= -I*dt;
                k4[j] += inputE(zj, currentT4)*dt;
                psi[j] += k1[j]/6.0+k2[j]/3.0+k3[j]/3.0+k4[j]/6.0;
            }
            for (int j = m_params.NAtoms; j < vec_size; ++j) {
                k4[j] *= -I*dt;
                psi[j] += k1[j]/6.0+k2[j]/3.0+k3[j]/3.0+k4[j]/6.0;
            }
            if (compute_electric_field) {
                std::complex<double> E_R_val;
                std::complex<double> E_L_val;
                std::complex<double> E_R_free_val;
                std::complex<double> E_L_free_val;
                if (m_params.use_input_electric_field_data) {
                    electricFieldRLFixedInput(E_R_val, E_L_val,
                                              E_R_free_val, E_L_free_val,
                                              psi, i);
                } else {
                    // Note that at this point we have already
                    // propagated in time by the step dt. Hence we
                    // have now the solution at time i*dt+dt=currentT4
                    // instead of i*dt=currentT1
                    electricFieldRL(E_R_val, E_L_val,
                                    E_R_free_val, E_L_free_val,
                                    psi, currentT4);
                }
                m_E_R(i) = E_R_val;
                if (restart_if_bogus_results) {
                    // It is enough to test only one of the
                    // fields, since they both get contribution
                    // from the spinwave, and it is the spinwave
                    // which turns NaN in the first place if the
                    // number of time steps is chosen too small.
                    if (std::isnan(m_E_R(i).real())) {
                        std::cout << " E_R(i=" << i << ") is NaN." << std::endl;
                        return false;
                    }
                }
                m_E_L(i) = E_L_val;
                m_E_R_free(i) = E_R_free_val;
                m_E_L_free(i) = E_L_free_val;
            }
            if (compute_spinwave_norm) {
                double normSquared = 0;
                double mean = 0;
                for (int j = 0; j < m_params.NAtoms; ++j) {
                    const double jNormSq = std::norm(psi[j+m_params.NAtoms]);
                    normSquared += jNormSq;
                    const double z = m_params.atom_positions[j];
                    mean += z*jNormSq;
                }
                if (normSquared > 1e-6) {
                    mean /= normSquared;
                } else {
                    mean = 0;
                }
                m_cStatesNorm(i) = std::sqrt(normSquared);
                m_cStatesMean(i) = mean;
            }
            if (compute_excited_norm) {
                double normSquared = 0;
                for (int j = 0; j < m_params.NAtoms; ++j) {
                    const double jNormSq = std::norm(psi[j]);
                    normSquared += jNormSq;
                }
                m_bStatesNorm(i) = std::sqrt(normSquared);
            }
        }
        if (store_intermediate_spinwave) {
            std::cout << "Intermediate value n = " << n << " / " << outer_loop_iterations << std::endl;
            m_psi_intermediate_snapshots.row(n) = psi;
        }
    }
    final = psi;
    m_tVals = Eigen::ArrayXd::Zero(m_E_R.size());
    for (int i = 0; i < N_t; ++i) {
        m_tVals(i) = (i+1)*dt;
    }
    return true;
}

bool
LambdaHamiltonian1ExcitationMatrix::evolveRKOnceRandomOptimized(
        VectorXcd &final, const VectorXcd &initial, double t, unsigned N_t, int flags)
{
    double dt = t/N_t;
    bool compute_electric_field = false;
    bool compute_spinwave_norm = false;
    bool compute_excited_norm = false;
    bool restart_if_bogus_results = false;
    bool store_intermediate_spinwave = false;
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_ELECTRIC_FIELD) {
        compute_electric_field = true;
    }
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_SPINWAVE_NORM) {
        compute_spinwave_norm = true;
    }
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_EXCITED_NORM) {
        compute_excited_norm = true;
    }
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_RESTART_IF_BOGUS_RESULTS) {
        restart_if_bogus_results = true;
    }
    const int vec_size = initial.size();
    assert(vec_size == 2*m_params.NAtoms
           && "State vector does not have the length 2*NAtoms!");
    if (flags & LAMBDA_HAMILTONIAN_EVOLVE_STORE_INTERMEDIATE_SPINWAVE_STATE)
    {
        store_intermediate_spinwave = true;
        m_psi_intermediate_snapshots
                = Eigen::MatrixXcd::Zero(
                    vec_size,
                    m_params.intermediate_spinwave_snapshots);
    }
    if (compute_electric_field) {
        m_E_R = VectorXcd::Zero(N_t);
        m_E_L = VectorXcd::Zero(N_t);
        m_E_R_free = VectorXcd::Zero(N_t);
        m_E_L_free = VectorXcd::Zero(N_t);
    } else {
        m_E_R = VectorXcd();
        m_E_L = VectorXcd();
        m_E_R_free = VectorXcd();
        m_E_L_free = VectorXcd();
    }
    if (compute_spinwave_norm) {
        m_cStatesNorm = Eigen::VectorXd::Zero(N_t);
        m_cStatesMean = Eigen::VectorXd::Zero(N_t);
    } else {
        m_cStatesNorm = Eigen::VectorXd();
        m_cStatesMean = Eigen::VectorXd();
    }
    if (compute_excited_norm) {
        m_bStatesNorm = Eigen::VectorXd::Zero(N_t);
    } else {
        m_bStatesNorm = Eigen::VectorXd();
    }
    if (m_OmegaTable.size() == 0) {
        m_OmegaTable = Eigen::VectorXcd(m_params.NAtoms);
    }
    const std::complex<double> I(0,1);
    const double a = 0.05;

    int outer_loop_iterations = 1;
    int inner_loop_iterations = N_t;
    if (store_intermediate_spinwave) {
        outer_loop_iterations = m_params.intermediate_spinwave_snapshots;
        // The assumption is that the caller of this function
        // has arranged for the number of the snapshots to
        // be a divisor of the total number of iterations
        inner_loop_iterations = N_t / m_params.intermediate_spinwave_snapshots;
    }
    VectorXcd partB = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd partC = VectorXcd::Zero(m_params.NAtoms);
    for (int i = 0; i < m_params.NAtoms; ++i) {
        partB(i) = initial(i);
        partC(i) = initial(i + m_params.NAtoms);
    }
    VectorXcd k1B = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd k1C = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd k2B = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd k2C = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd k3B = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd k3C = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd k4B = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd k4C = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd tempB = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd tempC = VectorXcd::Zero(m_params.NAtoms);
    VectorXcd OmegaTable = VectorXcd::Zero(m_params.NAtoms);
    for (int n = 0; n < outer_loop_iterations; ++n) {
        int i_min = 0;
        int i_max = N_t;
        if (store_intermediate_spinwave) {
            i_min = n*inner_loop_iterations;
            i_max = i_min + inner_loop_iterations;
        }
        for (int i = i_min; i < i_max; ++i) {
            //k1=-1i*H*psi0*Dt;
            //k2=-1i*H*(psi0+k1*0.5)*Dt;
            //k3=-1i*H*(psi0+k2*0.5)*Dt;
            //k4=-1i*H*(psi0+k3)*Dt;
            //psi0 = psi0 + (k1/6) + (k2/3) + (k3/3) + (k4/6);

            const double currentT1 = i*dt;
            const double currentT2 = currentT1+dt/2;
            const double currentT3 = currentT2;
            const double currentT4 = currentT1+dt;

            updateClassicalDrive(a, currentT1, t);
            OmegaTable = m_OmegaTable;
            k1B = m_expMatrixThreaded*partB;
            k1B += (-OmegaTable.array() * partC.array()).matrix();
            k1C = (-OmegaTable.array().conjugate() * partB.array()).matrix()
                  + (-m_params.delta * partC.array()).matrix();

            updateClassicalDrive(a, currentT2, t);
            OmegaTable = m_OmegaTable;
            for (int j = 0; j < m_params.NAtoms; ++j) {
                const double zj = m_params.atom_positions[j];
                k1B(j) *= -I*dt;
                k1B(j) += inputE(zj, currentT1)*dt;
                tempB(j) = partB(j) + 0.5*k1B(j);
            }
            for (int j = 0; j < m_params.NAtoms; ++j) {
                k1C(j) *= -I*dt;
                tempC(j) = partC(j) + 0.5*k1C(j);
            }
            k2B = m_expMatrixThreaded*tempB;
            k2B += (-OmegaTable.array() * tempC.array()).matrix();
            k2C = (-OmegaTable.array().conjugate() * tempB.array()).matrix()
                  + (-m_params.delta * tempC.array()).matrix();

            // The lines below are commented out
            // because currentT3 == currentT2

            //updateClassicalDrive(a, currentT3, t);
            //OmegaTable = m_OmegaTable;

            for (int j = 0; j < m_params.NAtoms; ++j) {
                const double zj = m_params.atom_positions[j];
                k2B(j) *= -I*dt;
                k2B(j) += inputE(zj, currentT2)*dt;
                tempB(j) = partB(j) + 0.5*k2B(j);
            }
            for (int j = 0; j < m_params.NAtoms; ++j) {
                k2C(j) *= -I*dt;
                tempC(j) = partC(j) + 0.5*k2C(j);
            }
            k3B = m_expMatrixThreaded*tempB;
            k3B += (-OmegaTable.array() * tempC.array()).matrix();
            k3C = (-OmegaTable.array().conjugate() * tempB.array()).matrix()
                  + (-m_params.delta * tempC.array()).matrix();

            updateClassicalDrive(a, currentT4, t);
            OmegaTable = m_OmegaTable;
            for (int j = 0; j < m_params.NAtoms; ++j) {
                const double zj = m_params.atom_positions[j];
                k3B(j) *= -I*dt;
                k3B(j) += inputE(zj, currentT3)*dt;
                tempB(j) = partB(j) + k3B(j);
            }
            for (int j = 0; j < m_params.NAtoms; ++j) {
                k3C(j) *= -I*dt;
                tempC(j) = partC(j) + k3C(j);
            }
            k4B = m_expMatrixThreaded*tempB;
            k4B += (-OmegaTable.array() * tempC.array()).matrix();
            k4C = (-OmegaTable.array().conjugate() * tempB.array()).matrix()
                  + (-m_params.delta * tempC.array()).matrix();

            for (int j = 0; j < m_params.NAtoms; ++j) {
                const double zj = m_params.atom_positions[j];
                k4B(j) *= -I*dt;
                k4B(j) += inputE(zj, currentT4)*dt;
                partB(j) += k1B(j)/6.0+k2B(j)/3.0+k3B(j)/3.0+k4B(j)/6.0;
            }
            for (int j = 0; j < m_params.NAtoms; ++j) {
                k4C(j) *= -I*dt;
                partC(j) += k1C(j)/6.0+k2C(j)/3.0+k3C(j)/3.0+k4C(j)/6.0;
            }
            if (compute_electric_field) {
                if (m_params.use_input_electric_field_data) {
                    std::complex<double> E_R_val(0,0);
                    std::complex<double> E_L_val(0,0);
                    std::complex<double> E_R_free_val(0,0);
                    std::complex<double> E_L_free_val(0,0);

                    E_R_val += partB.cwiseProduct(m_E_R_factors).sum();
                    E_L_val += partB.cwiseProduct(m_E_L_factors).sum();

                    E_R_val += m_E_R_input_data[n];
                    E_R_free_val += m_E_R_input_data[n];
                    E_L_val += m_E_L_input_data[n];
                    E_L_free_val += m_E_L_input_data[n];

                    m_E_R(i) = E_R_val;
                    m_E_L(i) = E_L_val;
                    m_E_R_free(i) = E_R_free_val;
                    m_E_L_free(i) = E_L_free_val;
                } else {
                    std::complex<double> E_R_val(0,0);
                    std::complex<double> E_L_val(0,0);
                    std::complex<double> E_R_free_val(0,0);
                    std::complex<double> E_L_free_val(0,0);

                    E_R_val += partB.cwiseProduct(m_E_R_factors).sum();
                    E_L_val += partB.cwiseProduct(m_E_L_factors).sum();

                    // The z argument is rescaled, so that it is
                    // actually z/L=1 and z/L=0 respectively.
                    //
                    // Also, note that at this point we have already
                    // propagated in time by the step dt. Hence we
                    // have now the solution at time i*dt+dt=currentT4
                    // instead of i*dt=currentT1
                    E_R_val += inputE_R(1, currentT4);
                    E_R_free_val += inputE_R(1, currentT4);
                    E_L_val += inputE_L(0, currentT4);
                    E_L_free_val += inputE_L(0, currentT4);

                    m_E_R(i) = E_R_val;
                    m_E_L(i) = E_L_val;
                    m_E_R_free(i) = E_R_free_val;
                    m_E_L_free(i) = E_L_free_val;
                }
                if (restart_if_bogus_results) {
                    // It is enough to test only one of the
                    // fields, since they both get contribution
                    // from the spinwave, and it is the spinwave
                    // which turns NaN in the first place if the
                    // number of time steps is chosen too small.
                    if (std::isnan(m_E_R(i).real())) {
                        std::cout << " E_R(i=" << i << ") is NaN." << std::endl;
                        return false;
                    }
                }
            }
            if (compute_spinwave_norm) {
                double normSquared = 0;
                double mean = 0;
                for (int j = 0; j < m_params.NAtoms; ++j) {
                    const double jNormSq = std::norm(partC(j));
                    normSquared += jNormSq;
                    const double z = m_params.atom_positions[j];
                    mean += z*jNormSq;
                }
                if (normSquared > 1e-6) {
                    mean /= normSquared;
                } else {
                    mean = 0;
                }
                m_cStatesNorm(i) = std::sqrt(normSquared);
                m_cStatesMean(i) = mean;
            }
            if (compute_excited_norm) {
                double normSquared = 0;
                for (int j = 0; j < m_params.NAtoms; ++j) {
                    const double jNormSq = std::norm(partB(j));
                    normSquared += jNormSq;
                }
                m_bStatesNorm(i) = std::sqrt(normSquared);
            }
        }
    }
    for (int i = 0; i < m_params.NAtoms; ++i) {
        final(i) = partB(i);
        final(i + m_params.NAtoms) = partC(i);
    }
    m_tVals = Eigen::ArrayXd::Zero(m_E_R.size());
    for (int i = 0; i < N_t; ++i) {
        m_tVals(i) = (i+1)*dt;
    }
    return true;
}

LambdaHamiltonian1Excitation::LambdaHamiltonian1Excitation() :
    d(new LambdaHamiltonian1ExcitationMatrix)
{
}

LambdaHamiltonian1Excitation::LambdaHamiltonian1Excitation(const HamiltonianParams &params) :

    d(new LambdaHamiltonian1ExcitationMatrix)
{
    d->setParams(params);
}

LambdaHamiltonian1Excitation::~LambdaHamiltonian1Excitation()
{
    delete d;
}

HamiltonianParams *LambdaHamiltonian1Excitation::params() const
{
    return &d->m_params;
}

void LambdaHamiltonian1Excitation::setParams(const HamiltonianParams &params)
{
    d->setParams(params);
}

Eigen::VectorXcd LambdaHamiltonian1Excitation::evolve(
        const Eigen::VectorXcd &initial, double t,
        EvolutionMethod method, int flags, double dt)
{
    EvolutionMethod evolutionMethod = method;
    if (evolutionMethod == EvolutionMethod::Default) {
        evolutionMethod = d->m_params.evolutionMethod;
    }
    switch (evolutionMethod) {
    case EvolutionMethod::RK4:
        return d->evolveRK(initial, t, dt, flags);
    case EvolutionMethod::MatrixExp:
        return d->evolveExp(initial, t);
    default:
        assert(0 && "Unknown evolution method!");
    }
}

std::vector<std::string> LambdaHamiltonian1Excitation::basisStates() const
{
    std::vector<std::string> ret;
    ret.reserve(d->m_NBase);
    for (int i = 0; i < d->m_params.NAtoms; ++i) {
        std::stringstream stateStream;
        stateStream << "b_" << i << " a";
        ret.emplace_back(stateStream.str());
    }
    for (int i = 0; i < d->m_params.NAtoms; ++i) {
        std::stringstream stateStream;
        stateStream << "c_" << i << " a";
        ret.emplace_back(stateStream.str());
    }
    return ret;
}

Eigen::VectorXcd LambdaHamiltonian1Excitation::E_R() const
{
    return d->E_R();
}

Eigen::VectorXcd LambdaHamiltonian1Excitation::E_L() const
{
    return d->E_L();
}

Eigen::VectorXcd LambdaHamiltonian1Excitation::E_R_free() const
{
    return d->E_R_free();
}

Eigen::VectorXcd LambdaHamiltonian1Excitation::E_L_free() const
{
    return d->E_L_free();
}

Eigen::VectorXd LambdaHamiltonian1Excitation::bStatesNorm() const
{
    return d->bStatesNorm();
}

Eigen::VectorXd LambdaHamiltonian1Excitation::cStatesNorm() const
{
    return d->cStatesNorm();
}

Eigen::VectorXd LambdaHamiltonian1Excitation::cStatesMean() const
{
    return d->cStatesMean();
}

Eigen::VectorXd LambdaHamiltonian1Excitation::tVals() const
{
    return d->tVals();
}

Eigen::MatrixXcd LambdaHamiltonian1Excitation::intermediateSolutions() const
{
    return d->intermediateSolutions();
}

int LambdaHamiltonian1Excitation::NBase() const
{
    return d->NBase();
}

