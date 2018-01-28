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

#ifndef URANDOM_H
#define URANDOM_H

#include <algorithm>
#include <random>
#include <cstdint>

typedef uint32_t URANDOMTYPE;

URANDOMTYPE getRandNumber();

template <typename Generator>
void generate_random_atom_positions(double *atom_positions, Generator &r,
                                    int NAtoms)
{
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < NAtoms; ++i) {
        atom_positions[i] = distribution(r);
    }
    std::sort(&atom_positions[0], (&atom_positions[0])+NAtoms);
    // We always put an atom at the left edge of the
    // ensemble. This way the random placement matches
    // the regular placement more closely.
    const double positionShift = atom_positions[0];
    for (int i = 0; i < NAtoms; ++i) {
        atom_positions[i] -= positionShift;
    }
}

template <typename Generator>
void generate_atom_positions_distributed_around_regular_spacing(
        double *atom_positions, Generator r, int NAtoms, double width)
{
    const double rescaledWidth = width/NAtoms;
    const double meanStep = 1.0/NAtoms;
    if (width == 0) {
        for (int i = 0; i < NAtoms; ++i) {
            atom_positions[i] = i*meanStep;
        }
    } else {
        //std::normal_distribution<double> distribution(0.0, rescaledWidth);
        std::uniform_real_distribution<double> distribution(0.0, rescaledWidth);
        for (int i = 0; i < NAtoms; ++i) {
            const double gaussian_shift = distribution(r);
            //printf("Gaussian shift %d: %.16f\n", i, gaussian_shift);
            atom_positions[i] = gaussian_shift + i*meanStep;
        }
    }
    // For small widths we probably don't need
    // any sorting but for the general case we
    // certainly do.
    std::sort(&atom_positions[0], (&atom_positions[0])+NAtoms);
}

#endif // URANDOM_H

