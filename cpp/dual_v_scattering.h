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

#ifndef DUAL_V_SCATTERING_H
#define DUAL_V_SCATTERING_H

#include "Eigen/Dense"

inline void dualv_scattering_coefficients(
        std::complex<double> &rpp, std::complex<double> &tpp,
        std::complex<double> &rmp, std::complex<double> &tmp,
        std::complex<double> &rpm, std::complex<double> &tpm,
        std::complex<double> &rmm, std::complex<double> &tmm,
        double Deltac_p, double Deltac_m,
        std::complex<double> delta, double g1d, double Omega_p, double Omega_m,
        std::complex<double> OmegaPhaseFactor)
{
    const std::complex<double> I(0,1);
    const std::complex<double> Delta_p = Deltac_p + delta + 0.5*I;
    const std::complex<double> Delta_m = Deltac_m + delta + 0.5*I;
    const double Omega_p2 = Omega_p*Omega_p;
    const double Omega_m2 = Omega_m*Omega_m;
    const std::complex<double> denominator
            = Delta_p*Delta_m*delta-Delta_p*Omega_m2-Delta_m*Omega_p2;
    rpp = -0.5*I*g1d*(Delta_m*delta-Omega_m2)/denominator;
    tpp = 1.0+rpp;
    rpm = -0.5*I*g1d*Omega_m*Omega_p*std::conj(OmegaPhaseFactor)/denominator;
    tpm = rpm;
    rmm = -0.5*I*g1d*(Delta_p*delta-Omega_p2)/denominator;
    tmm = 1.0+rmm;
    rmp = -0.5*I*g1d*Omega_p*Omega_m*OmegaPhaseFactor/denominator;
    tmp = rmp;
}

inline Eigen::Matrix4cd scattering_matrix4_beta(std::complex<double> rpp,
                                                std::complex<double> rmp,
                                                std::complex<double> rpm,
                                                std::complex<double> rmm)
{
    Eigen::Matrix2cd S_r;
    S_r(0,0) = rpp;
    S_r(0,1) = rmp;
    S_r(1,0) = rpm;
    S_r(1,1) = rmm;

    Eigen::Matrix2cd S_t = Eigen::Matrix2cd::Identity() + S_r;

    Eigen::FullPivLU<Eigen::Matrix2cd> lu(S_t);
    Eigen::Matrix2cd B = -lu.solve(S_r);

    Eigen::Matrix4cd ret;

    ret.block<2,2>(0,0) = Eigen::Matrix2cd::Identity() - B;
    ret.block<2,2>(0,2) = -B;
    ret.block<2,2>(2,0) = B;
    ret.block<2,2>(2,2) = Eigen::Matrix2cd::Identity() + B;
    return ret;
}

#endif // DUAL_V_SCATTERING_H

