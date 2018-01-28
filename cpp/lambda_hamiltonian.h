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

#ifndef LAMBDA_HAMILTONIAN_H
#define LAMBDA_HAMILTONIAN_H

#include <string>
#include <vector>

#include "Eigen/Dense"

#include "hamiltonian_params.h"

#define LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_ELECTRIC_FIELD          (1 << 0)
#define LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_SPINWAVE_NORM           (1 << 1)
#define LAMBDA_HAMILTONIAN_EVOLVE_RESTART_IF_BOGUS_RESULTS          (1 << 2)
#define LAMBDA_HAMILTONIAN_EVOLVE_STORE_INTERMEDIATE_SPINWAVE_STATE (1 << 3)
#define LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_EXCITED_NORM            (1 << 1)

class LambdaHamiltonian1ExcitationMatrix;

class LambdaHamiltonian1Excitation
{
public:
    LambdaHamiltonian1Excitation();
    LambdaHamiltonian1Excitation(const HamiltonianParams &params);
    ~LambdaHamiltonian1Excitation();
    HamiltonianParams *params() const;
    void setParams(const HamiltonianParams &params);
    Eigen::VectorXcd evolve(
            const Eigen::VectorXcd &initial,
            double t,
            EvolutionMethod method = EvolutionMethod::Default,
            int flags = 0,
            double dt = 0.04);
    Eigen::VectorXcd evolveExp(const Eigen::VectorXcd &initial, double t)
    {
        return evolve(initial, t, EvolutionMethod::MatrixExp);
    }
    Eigen::VectorXcd E_R() const;
    Eigen::VectorXcd E_L() const;
    Eigen::VectorXcd E_R_free() const;
    Eigen::VectorXcd E_L_free() const;
    Eigen::VectorXd bStatesNorm() const;
    Eigen::VectorXd cStatesNorm() const;
    Eigen::VectorXd cStatesMean() const;
    Eigen::VectorXd tVals() const;
    Eigen::MatrixXcd intermediateSolutions() const;
    int NBase() const;
    std::vector<std::string> basisStates() const;

private:
    LambdaHamiltonian1ExcitationMatrix *const d;
};

#endif // LAMBDA_HAMILTONIAN_H

