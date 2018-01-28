#!/usr/bin/python

# Copyright (c) 2017 Ivan Iakoupov
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os.path

import pylab as p
import numpy as np
import plot_dispersion_relations_common as pdrc
from matplotlib_util import extract_params_from_file_name
from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

def quadratic_dispersion_relation(N_u, Delta3, Deltac, Omega, g1d):
    gprime = 1-g1d
    m = -(N_u-1)*g1d**2/(N_u**2*(2*Deltac+1j*gprime)*Omega**2)
    qd = np.sqrt(2*m*Delta3)
    return qd

g1d = 0.1
Deltac = -90
Omega = 1

def draw_annotation(axis, n, f, text, text_x):
    ax.annotate(text,
                xy=(q_array_re_Lambda_cold_list[f][n], delta_array2[n]), xycoords='data',
                xytext=(text_x, delta_array2[n]), textcoords='data',
                size=10, va="center", ha="left",
                arrowprops=dict(arrowstyle="->",
                                shrinkA=0,
                                shrinkB=0,
                                connectionstyle="arc3,rad=0"),
    )

def draw_annotation_vertical(axis, n, f, text, text_y):
    ax.annotate(text,
                xy=(q_array_re_Lambda_cold_list[f][n], delta_array2[n]), xycoords='data',
                xytext=(q_array_re_Lambda_cold_list[f][n], text_y), textcoords='data',
                size=10, va="center", ha="center",
                arrowprops=dict(arrowstyle="->",
                                shrinkA=0,
                                shrinkB=0,
                                connectionstyle="arc3,rad=0"),
    )


usetex()
p.figure(figsize=(3.3,2.5))
ax = p.subplot(111)

NFourier = 6
delta_array_log = np.linspace(-6, 0, 500)
delta_array2 = 10**delta_array_log
delta_array2_len = len(delta_array2)
q_array_re_Lambda_cold_analytical = np.zeros_like(delta_array2)
q_array_im_Lambda_cold_analytical = np.zeros_like(delta_array2)
q_array_re_Lambda_cold_list = []
q_array_im_Lambda_cold_list = []
for f in range(2*NFourier):
    q_array_re_Lambda_cold_list.append(np.zeros_like(delta_array2))
    q_array_im_Lambda_cold_list.append(np.zeros_like(delta_array2))

q_array_re_dualV = np.zeros_like(delta_array2)
q_array_im_dualV = np.zeros_like(delta_array2)
q_array_re_eit = np.zeros_like(delta_array2)
q_array_im_eit = np.zeros_like(delta_array2)
for n in range(delta_array2_len):
    q_n_Lambda_cold_analytical = pdrc.q_Lambda_cold(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_Lambda_cold_analytical[n] = q_n_Lambda_cold_analytical.real
    q_array_im_Lambda_cold_analytical[n] = q_n_Lambda_cold_analytical.imag

    phaseShift_1 = 0
    for f in range(NFourier):
        phaseShift_2 = 2**(-(f+2))
        q_n_Lambda_cold_1 = np.absolute(pdrc.q_eit_standing_wave(delta_array2[n]+Deltac, Deltac, Omega, g1d, 2**(f+1), phaseShift_1))
        q_array_re_Lambda_cold_list[2*f][n] = q_n_Lambda_cold_1.real
        q_array_im_Lambda_cold_list[2*f][n] = q_n_Lambda_cold_1.imag

        q_n_Lambda_cold_2 = np.absolute(pdrc.q_eit_standing_wave(delta_array2[n]+Deltac, Deltac, Omega, g1d, 2**(f+1), phaseShift_2))
        q_array_re_Lambda_cold_list[2*f+1][n] = q_n_Lambda_cold_2.real
        q_array_im_Lambda_cold_list[2*f+1][n] = q_n_Lambda_cold_2.imag

    q_n_dualV = pdrc.q_secular(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_dualV[n] = q_n_dualV.real
    q_array_im_dualV[n] = q_n_dualV.imag

    q_n_eit = pdrc.q_eit(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_eit[n] = q_n_eit.real
    q_array_im_eit[n] = q_n_eit.imag

usetex()

p.loglog(q_array_re_eit, delta_array2, 'k-',
         label='EIT',
         linewidth=common_line_width)
for f in range(NFourier):
    p.loglog(q_array_re_Lambda_cold_list[2*f], delta_array2, 'g--',
             label='$\Lambda$-type n',
             linewidth=common_line_width)
    p.loglog(q_array_re_Lambda_cold_list[2*f+1], delta_array2, 'm-.',
             label='$\Lambda$-type n',
             linewidth=common_line_width)
p.loglog(q_array_re_Lambda_cold_analytical, delta_array2, 'b-',
         label='$\Lambda$-type',
         linewidth=common_line_width)
p.loglog(np.absolute(q_array_re_dualV), delta_array2, 'r-',
         label='dual-V (secular)',
         linewidth=common_line_width)

common_x = 1.9e-4
for f in range(5):
    draw_annotation(ax, 180-f*35, 2*f, r'$N_{{\rm u}}={}$'.format(2**(1+f)), common_x)

draw_annotation_vertical(ax, 240, 1, r'$N_{{\rm u}}={}$'.format(2**(1+0)), 1e-1)
draw_annotation_vertical(ax, 195, 3, r'$N_{{\rm u}}={}$'.format(2**(1+1)), 3e-2)
draw_annotation_vertical(ax, 150, 5, r'$N_{{\rm u}}={}$'.format(2**(1+2)), 1e-2)
draw_annotation_vertical(ax, 105, 7, r'$N_{{\rm u}}={}$'.format(2**(1+3)), 3e-3)
draw_annotation_vertical(ax, 60, 9, r'$N_{{\rm u}}={}$'.format(2**(1+4)), 1e-3)

p.xlabel(r'${\rm Re}[q]/n_0$')
p.ylabel(r"$\delta/\Gamma$")
p.xlim(1e-6,1e-3)
p.tight_layout(pad=0.2)

p.savefig('fig5.eps')
