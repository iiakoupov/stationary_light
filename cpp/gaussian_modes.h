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

#ifndef GAUSSIAN_MODES_H
#define GAUSSIAN_MODES_H

#include <complex>

inline double gaussian_mode(double z, double width)
{
    const double A = std::pow(2*M_PI, 0.25);
    return 1.0/(A*std::sqrt(width))*exp(-std::pow(z/width, 2)/4);
}

inline std::complex<double> gaussian_final(double z, double sigma,
                                           double t,
                                           std::complex<double> omega1,
                                           std::complex<double> omega2)
{
    const std::complex<double> I(0,1);
    return 1/(std::pow(2*M_PI, 0.25)*std::sqrt(sigma))
           *std::exp(-std::pow((z-omega1*t), 2)/(4.0*(std::pow(sigma, 2)+I*omega2*t/2.0)))
           *std::sqrt(1.0/(1.0+I*omega2*t/(2*std::pow(sigma,2))));
}
#endif // GAUSSIAN_MODES_H
