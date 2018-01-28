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

#ifndef ENSEMBLE_SCATTERING_COMMON_H
#define ENSEMBLE_SCATTERING_COMMON_H

#include <complex>
#include "Eigen/Dense"

#define REFLECTION_COEFFICIENT_SCATTERING_FROM_THE_LEFT

struct RandTCoefficients
{
    std::complex<double> t;
    std::complex<double> r;
    std::complex<double> t_impurity;
    std::complex<double> r_impurity;
    std::complex<double> t_damaged;
    std::complex<double> r_damaged;
    std::complex<double> t_damaged_impurity;
    std::complex<double> r_damaged_impurity;
    std::complex<double> r_empty_cavity;
    std::complex<double> t_empty_cavity;
    std::complex<double> r_regular_eit;
    std::complex<double> t_regular_eit;
    std::complex<double> r_damaged_regular_eit;
    std::complex<double> t_damaged_regular_eit;
    Eigen::MatrixXcd Mensemble;
    Eigen::MatrixXcd Mensemble_impurity;
};

#endif // ENSEMBLE_SCATTERING_COMMON_H

