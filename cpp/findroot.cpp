/* Copyright (c) 2015 Ivan Iakoupov
 *
 * Based in part upon code from mpmath which is:
 *
 * Copyright (c) 2005-2010 Fredrik Johansson and mpmath contributors
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  a. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  b. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *  c. Neither the name of mpmath nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

#include "findroot.h"

std::complex<double>
find_root_muller(std::function<std::complex<double>(std::complex<double>)> f,
                 std::complex<double> x0, std::complex<double> x1,
                 std::complex<double> x2, double tol)
{
    /*
    mpmath implementation:

    fx0 = f(x0)
    fx1 = f(x1)
    fx2 = f(x2)
    while True:
        # TODO: maybe refactoring with function for divided differences
        # calculate divided differences
        fx2x1 = (fx1 - fx2) / (x1 - x2)
        fx2x0 = (fx0 - fx2) / (x0 - x2)
        fx1x0 = (fx0 - fx1) / (x0 - x1)
        w = fx2x1 + fx2x0 - fx1x0
        fx2x1x0 = (fx1x0 - fx2x1) / (x0 - x2)
        if w == 0 and fx2x1x0 == 0:
            if self.verbose:
                print('canceled with')
                print('x0 =', x0, ', x1 =', x1, 'and x2 =', x2)
            break
        x0 = x1
        fx0 = fx1
        x1 = x2
        fx1 = fx2
        # denominator should be as large as possible => choose sign
        r = self.ctx.sqrt(w**2 - 4*fx2*fx2x1x0)
        if abs(w - r) > abs(w + r):
            r = -r
        x2 -= 2*fx2 / (w + r)
        fx2 = f(x2)
        error = abs(x2 - x1)
        yield x2, error
    */
    std::complex<double> tempX0 = x0;
    std::complex<double> tempX1 = x1;
    std::complex<double> tempX2 = x2;
    std::complex<double> fx0 = f(tempX0);
    std::complex<double> fx1 = f(tempX1);
    std::complex<double> fx2 = f(tempX2);
    double abserr;
    for (int i = 0; i < 500; ++i) {
        const std::complex<double> fx2x1 = (fx1 - fx2)/(tempX1 - tempX2);
        const std::complex<double> fx2x0 = (fx0 - fx2)/(tempX0 - tempX2);
        const std::complex<double> fx1x0 = (fx0 - fx1)/(tempX0 - tempX1);
        const std::complex<double> w = fx2x1 + fx2x0 - fx1x0;
        const std::complex<double> fx2x1x0 = (fx1x0 - fx2x1)/(tempX0 - tempX2);
        if (w == 0.0 and fx2x1x0 == 0.0) {
            break;
        }
        tempX0 = tempX1;
        fx0 = fx1;
        tempX1 = tempX2;
        fx1 = fx2;
        // denominator should be as large as possible => choose sign
        std::complex<double> r = std::sqrt(w*w - 4.0*fx2*fx2x1x0);
        if (std::abs(w-r) > std::abs(w+r)) {
            r = -r;
        }
        tempX2 -= 2.0*fx2/(w+r);
        fx2 = f(tempX2);
        abserr = std::abs(tempX2 - tempX1);
        if (abserr < tol) {
            break;
        }
    }

    return tempX2;
}

