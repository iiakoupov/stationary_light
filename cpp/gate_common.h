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

#ifndef GATE_COMMON_H
#define GATE_COMMON_H

#include <cmath>
#include "quadratic_equation.h"

class OptimalDeltaFromDispersionRelation
{
    double m_Deltac;
    double m_OmegaScattering;
    double m_g1d;
    int m_periodLength;
    int m_NAtoms;
public:
    OptimalDeltaFromDispersionRelation(double Deltac, double OmegaScattering,
                                       double g1d, int periodLength,
                                       int NAtoms);

    double operator() (double Delta1);
};

inline int next_power_of_two(int n)
{
    int ret = 1;
    while (ret < n) {
        ret *= 2;
    }
    return ret;
}

inline void approximate_resonance_solution1_coefficients(
        double &a, double &b, double &c,
        double g1d, double NAtoms, double Deltac, double Omega)
{
    // Here we use the formula as for the 3/2*pi (1/2*pi) setup.
    // We use the same formula even in the case of 5/4*pi, 9/8*pi etc.
    // For the other setups (especially the ones approaching the random
    // limit) the returned value will only serve as an order of magnitude
    // guess and will need to be refined further.
    const int n = NAtoms/2;
    const int k = n-1;

    const double DeltacPow2 = Deltac*Deltac;
    const double DeltacPow4 = DeltacPow2*DeltacPow2;
    const double g1dPow2 = g1d*g1d;
    const double gprime = 1-g1d;
    const double gprimePow2 = gprime*gprime;
    const double gprimePow4 = gprimePow2*gprimePow2;
    const double OmegaPow2 = Omega*Omega;
    const double OmegaPow4 = OmegaPow2*OmegaPow2;

    a = (g1dPow2*(16*DeltacPow4+8*DeltacPow2*gprimePow2+gprimePow4
                               -16*DeltacPow2*OmegaPow2+4*gprimePow2*OmegaPow2))
            /(2*std::pow(4*DeltacPow2 + gprimePow2,2)*OmegaPow4);
    b = (2*Deltac*g1dPow2)/((4*DeltacPow2+gprimePow2)*OmegaPow2);
    c = -(-1-std::cos(M_PI*double(k)/n));
}

inline double
approximate_resonance_solution1_discriminant(double g1d, double NAtoms,
                                             double Deltac, double Omega)
{
    double a;
    double b;
    double c;
    approximate_resonance_solution1_coefficients(a, b, c, g1d, NAtoms, Deltac,
                                                 Omega);
    const double Discriminant = b*b-4*a*c;
    return Discriminant;
}

inline double
approximate_resonance_solution1(double g1d, double NAtoms, double Deltac, double Omega)
{
    double a;
    double b;
    double c;
    approximate_resonance_solution1_coefficients(a, b, c, g1d, NAtoms, Deltac,
                                                 Omega);
    return quadratic_equation_root1(a, b, c);
}
#endif // GATE_COMMON_H

