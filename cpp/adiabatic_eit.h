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

#ifndef ADIABATIC_EIT_H
#define ADIABATIC_EIT_H

#include <complex>
#include "quadrature/quad.h"

template <class F>
std::complex<double>
adiabatic_eit_retrieve_electric_field_from_spinwave_asymptotic(
        F spinwave, double t, double g1d, double NAtoms, double delta,
        double Deltac, double Omega, double tolAbs = 1e-7, double tolRel = 0.1)
{
    const std::complex<double> I(0,1);
    const double Delta = Deltac+delta;
    const std::complex<double> gprimeDelta = (1-g1d)/2-I*Delta;
    const double g1dN = g1d/2*NAtoms;
    const double Omega2 = Omega*Omega;
    const std::complex<double> ret
            = -std::exp(-I*delta*t)*std::sqrt(g1dN)*Omega/gprimeDelta*std::sqrt(gprimeDelta)/(2.0*std::sqrt(M_PI))
            *quad_tanh_sinh<std::complex<double>>([=] (double z) -> std::complex<double> {
        if (t == 0 || z == 0) {
            return 0;
        }
        const std::complex<double> exp_factor
                = 1.0/std::pow(Omega2*t*g1dN*(1-z), 0.25)
                  *std::exp(-std::pow(std::sqrt(Omega2*t)-std::sqrt(g1dN*(1-z)),2)/gprimeDelta);
        return exp_factor*spinwave(z);
    }, 0, 1, tolAbs, tolRel);
    return ret;
}

template <class F>
std::complex<double>
adiabatic_eit_store_spinwave_from_electric_field_asymptotic(
        F field, double z, double t, double g1d, double NAtoms, double delta,
        double Deltac, double Omega, double absTol, double relTol,
        double *absErr = 0, double *relErr = 0)
{
    const std::complex<double> I(0,1);
    const double Delta = Deltac+delta;
    const std::complex<double> gprimeDelta = (1-g1d)/2-I*Delta;
    const double g1dN = g1d/2*NAtoms;
    const double Omega2 = Omega*Omega;
    // Note that in principle the times are the rescaled
    // ones here: \tilde{t}=t-z/c. However, in the integrand
    // below, there is always a difference of two such times
    // and hence the constant z/c terms will cancel.
    const std::complex<double> ret
            = -std::sqrt(g1dN)*Omega/gprimeDelta*std::sqrt(gprimeDelta)/(2.0*std::sqrt(M_PI))
            *quad_tanh_sinh<std::complex<double>>([=] (double t1) -> std::complex<double> {
        if (t==t1 || z == 0) {
            return 0;
        }
        const std::complex<double> exp_factor
                = std::exp(I*delta*(t-t1))/std::pow(Omega2*(t-t1)*g1dN*z, 0.25)
                  *std::exp(-std::pow(std::sqrt(Omega2*(t-t1))-std::sqrt(g1dN*z),2)/gprimeDelta);
        return exp_factor*field(t1);
    }, 0, t, absTol, relTol, absErr, relErr, QUAD_BOTH_BOUNDS_FINITE, 18);
    return ret;
}

#endif // ADIABATIC_EIT_H
