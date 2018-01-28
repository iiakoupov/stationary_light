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

#ifndef QUADRATIC_EQUATION_H
#define QUADRATIC_EQUATION_H

// See http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
inline double quadratic_equation_root1(double a, double b, double c)
{
    if (b < 0) {
        return 2*c/(-b+std::sqrt(b*b-4*a*c));
    } else {
        return (-b-std::sqrt(b*b-4*a*c))/(2*a);
    }
}

inline double quadratic_equation_root2(double a, double b, double c)
{
    if (b < 0) {
        return (-b+std::sqrt(b*b-4*a*c))/(2*a);
    } else {
        return 2*c/(-b-std::sqrt(b*b-4*a*c));
    }
}

#endif // QUADRATIC_EQUATION
