#!/usr/bin/env python

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

import matplotlib.pyplot as p
import numpy as np
import plot_dispersion_relations_common as pdrc
from matplotlib_util import extract_params_from_file_name
from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

def q_Lambda_cold_approx(delta, g1d, Deltac, Omega):
    gprime = 1-g1d
    return np.sqrt(g1d**2*delta**(3.0/2)/(np.sqrt(-Deltac-0.5j*gprime)*Omega**3))

def delta_Lambda_cold_approx(q, g1d, Deltac, Omega):
    gprime = 1-g1d
    return (-Deltac-0.5j*gprime)**(1.0/3)*Omega**2/(g1d**(4.0/3))*q**(4.0/3)

def q_two_level_dispersion_relation(delta, g1d, Deltac):
    return -g1d/(2*(delta+Deltac))

g1d = 0.1
Deltac = -90
Deltad = 1
Omega = 1

NFourier = 7
delta_array_log = np.linspace(-6, 0, 500)
delta_array2 = 10**delta_array_log
delta_array2_len = len(delta_array2)
q_array_dependent_var_log = np.linspace(-6, -3, 500)
q_array_dependent_var = 10**q_array_dependent_var_log
q_array_dependent_var_len = len(q_array_dependent_var)

delta_array_re_Lambda_cold_analytical_approx = np.zeros_like(q_array_dependent_var)
delta_array_im_Lambda_cold_analytical_approx = np.zeros_like(q_array_dependent_var)
for n in range(q_array_dependent_var_len):
    delta_n_Lambda_cold_approx = delta_Lambda_cold_approx(q_array_dependent_var[n], g1d, Deltac, Omega)
    delta_array_re_Lambda_cold_analytical_approx[n] = delta_n_Lambda_cold_approx.real
    delta_array_im_Lambda_cold_analytical_approx[n] = delta_n_Lambda_cold_approx.imag

q_array_re_Lambda_cold_analytical = np.zeros_like(delta_array2)
q_array_im_Lambda_cold_analytical = np.zeros_like(delta_array2)
q_array_re_Lambda_cold_analytical_approx = np.zeros_like(delta_array2)
q_array_im_Lambda_cold_analytical_approx = np.zeros_like(delta_array2)
#q_array_re_Lambda_cold_big_nf = np.zeros_like(delta_array2)
#q_array_im_Lambda_cold_big_nf = np.zeros_like(delta_array2)
q_array_re_dual_color_1 = np.zeros_like(delta_array2)
q_array_im_dual_color_1 = np.zeros_like(delta_array2)
q_array_re_two_level = np.zeros_like(delta_array2)
q_array_im_two_level = np.zeros_like(delta_array2)
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
    q_n_Lambda_cold_analytical_approx = q_Lambda_cold_approx(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_Lambda_cold_analytical[n] = q_n_Lambda_cold_analytical.real
    q_array_im_Lambda_cold_analytical[n] = q_n_Lambda_cold_analytical.imag
    q_array_re_Lambda_cold_analytical_approx[n] = q_n_Lambda_cold_analytical_approx.real
    q_array_im_Lambda_cold_analytical_approx[n] = q_n_Lambda_cold_analytical_approx.imag

    #q_n_Lambda_cold_big_nf = pdrc.q_Lambda_cold_numeric(delta_array2[n], g1d,
    #                                                    Deltac, Omega, 1000)
    #q_array_re_Lambda_cold_big_nf[n] = q_n_Lambda_cold_big_nf.real
    #q_array_im_Lambda_cold_big_nf[n] = q_n_Lambda_cold_big_nf.imag
    q_n_dual_color_1_plus, q_n_dual_color_1_minus = pdrc.q_dual_color_numeric(delta_array2[n], g1d, Deltac,
                                                 Deltad, Omega, 100)
    q_array_re_dual_color_1[n] = q_n_dual_color_1_plus.real
    q_array_im_dual_color_1[n] = q_n_dual_color_1_plus.imag
    for f in range(NFourier):
        q_n_Lambda_cold_1 = pdrc.q_Lambda_cold_numeric(delta_array2[n], g1d, Deltac, Omega, 2**(f+1))
        q_n_Lambda_cold_2 = pdrc.q_Lambda_cold_numeric(delta_array2[n], g1d, Deltac, Omega, 2**(f+1)+1)
        q_array_re_Lambda_cold_list[2*f][n] = q_n_Lambda_cold_1.real
        q_array_im_Lambda_cold_list[2*f][n] = q_n_Lambda_cold_1.imag
        q_array_re_Lambda_cold_list[2*f+1][n] = q_n_Lambda_cold_2.real
        q_array_im_Lambda_cold_list[2*f+1][n] = q_n_Lambda_cold_2.imag

    q_n_dualV = pdrc.q_secular(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_dualV[n] = q_n_dualV.real
    q_array_im_dualV[n] = q_n_dualV.imag

    q_n_eit = pdrc.q_eit(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_eit[n] = q_n_eit.real
    q_array_im_eit[n] = q_n_eit.imag

    q_n_two_level = q_two_level_dispersion_relation(delta_array2[n], g1d, Deltac)
    q_array_re_two_level[n] = q_n_two_level.real
    q_array_im_two_level[n] = q_n_two_level.imag

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

p.loglog(q_array_re_eit, delta_array2, 'k-',
         label='EIT',
         linewidth=common_line_width)
for f in range(2*NFourier):
    p.loglog(q_array_re_Lambda_cold_list[f], delta_array2, 'g--',
             label='$\Lambda$-type n',
             linewidth=common_line_width)
p.loglog(q_array_re_Lambda_cold_analytical, delta_array2, 'b-',
         label='$\Lambda$-type',
         linewidth=common_line_width)

#The commented out plots are to check the
#approximate expression given in the paper

#p.loglog(q_array_re_Lambda_cold_analytical_approx, delta_array2, 'm--',
#         label='$\Lambda$-type (approx)',
#         linewidth=common_line_width)
#p.loglog(q_array_dependent_var, delta_array_re_Lambda_cold_analytical_approx, 'c--',
#         label='$\Lambda$-type (approx,alt)',
#         linewidth=common_line_width)

p.loglog(q_array_re_dualV, delta_array2, 'r-',
         label='dual-V (secular)',
         linewidth=common_line_width)
p.loglog(q_array_re_dual_color_1, delta_array2, 'c--',
         label='dual-color (Deltad/Gamma=1)',
         linewidth=common_line_width)

#p.loglog(q_array_re_two_level, delta_array2, 'k--',
#         label='two-level',
#         linewidth=common_line_width)

common_x = 1.9e-4

ax.annotate(r'$n=1$',
            xy=(q_array_re_dualV[210], delta_array2[210]), xycoords='data',
            xytext=(common_x, delta_array2[210]), textcoords='data',
            size=10, va="center", ha="left",
            arrowprops=dict(arrowstyle="->",
                            shrinkA=0,
                            shrinkB=0,
                            connectionstyle="arc3,rad=0"),
)
for f in range(5):
    draw_annotation(ax, 180-f*30, 2*f+1, r'$n={}$'.format(2**(1+f)+1), common_x)

draw_annotation_vertical(ax, 240, 0, r'$n={}$'.format(2**(1+0)), 1e-1)
draw_annotation_vertical(ax, 195, 2, r'$n={}$'.format(2**(1+1)), 3e-2)
draw_annotation_vertical(ax, 150, 4, r'$n={}$'.format(2**(1+2)), 1e-2)
draw_annotation_vertical(ax, 105, 6, r'$n={}$'.format(2**(1+3)), 3e-3)
draw_annotation_vertical(ax, 60, 8, r'$n={}$'.format(2**(1+4)), 1e-3)


p.xlabel(r'${\rm Re}[q]/n_0$')
p.ylabel(r"$\delta/\Gamma$")
p.xlim(1e-6, 1e-3)
p.ylim(1e-6, 1)
p.tight_layout(pad=0.2)

p.savefig('fig2.eps')
