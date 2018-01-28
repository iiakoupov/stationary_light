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

#include "gate_common.h"
#include "bandstructure.h"

OptimalDeltaFromDispersionRelation::OptimalDeltaFromDispersionRelation(
        double Deltac, double OmegaScattering, double g1d, int periodLength,
        int NAtoms) :
    m_Deltac(Deltac),
    m_OmegaScattering(OmegaScattering),
    m_g1d(g1d),
    m_periodLength(periodLength),
    m_NAtoms(NAtoms)
    {}

double OptimalDeltaFromDispersionRelation::operator() (double Delta3)
{
    double qd_re_abs;
    if (Delta3 > 0) {
        qd_re_abs = std::abs(qd_eit_standing_wave(
                             Delta3, m_Deltac, m_OmegaScattering,
                             m_g1d, 1-m_g1d, m_periodLength, 0).real());
    } else {
        // Here we replace the function values for the
        // negative values of Delta3 with the
        // mirror of it through the origin (so that
        // the function becomes odd). The function will look
        // something like this:
        //
        //                      -------
        //       Original  ->  /                   (qd_re_abs > 0)
        //                     |
        //   -------------------------------------- <- qd_re_abs == 0
        //                     |
        //                     /  <- Odd extension (qd_re_abs < 0)
        //              -------
        // We use this class for rootfinding of some
        // positive Delta3 for some positive value of qd_re_abs.
        // However, during the course of rootfinding it can
        // happen that we get some intermediate negative values of
        // Delta3. Such an odd function extension will make sure
        // that in this case qd_re_abs is negative and continuously
        // connects to the actual part of the function. The negative
        // value of qd_re_abs will natually redirect the rootfinding
        // algorithm back to the region of the positive Delta3.

        qd_re_abs = -std::abs(qd_eit_standing_wave(
                              -Delta3, m_Deltac, m_OmegaScattering,
                              m_g1d, 1-m_g1d, m_periodLength, 0).real());

    }
    return qd_re_abs-M_PI/m_NAtoms;
}

