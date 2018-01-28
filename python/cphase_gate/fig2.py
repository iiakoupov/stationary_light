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

import scipy.io as sio
import pylab as p
import numpy as np
import random
from r_and_t_plot_common import ensemble_transfer_matrix_pi_half,\
                                impurity_unit_cell_pi_half

from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

def plot_r_and_t(min_Delta3, max_Delta3, num_Delta3, NAtoms, kd, g1d, gprime,
                 gm, Deltap, Omega):
    Delta3Values = np.linspace(min_Delta3, max_Delta3, num_Delta3)
    r = np.zeros(num_Delta3, dtype=np.float64)
    t = np.zeros(num_Delta3, dtype=np.float64)
    r_imp = np.zeros(num_Delta3, dtype=np.float64)
    t_imp = np.zeros(num_Delta3, dtype=np.float64)
    r_min = 1
    Delta3_for_r_min = 0
    for n, Delta3 in enumerate(Delta3Values):
        Delta1 = Delta3+Deltap
        MensembleStandingWave\
                = ensemble_transfer_matrix_pi_half(NAtoms, g1d, gprime, gm, Delta1,
                                           Deltap, Omega)
        MHalfEnsembleStandingWave\
                = ensemble_transfer_matrix_pi_half(NAtoms/2, g1d, gprime, gm, Delta1,
                                           Deltap, Omega)
        M_impurity_cell = impurity_unit_cell_pi_half(g1d, gprime, gm, Delta1, Deltap, Omega)
        MensembleStandingWaveImpurity = MHalfEnsembleStandingWave*M_impurity_cell*MHalfEnsembleStandingWave
        r[n] = np.abs(-MensembleStandingWave[1,0]\
                      /MensembleStandingWave[1,1])**2
        t[n] = np.abs(1.0/MensembleStandingWave[1,1])**2
        r_imp[n] = np.abs(-MensembleStandingWaveImpurity[1,0]\
                          /MensembleStandingWaveImpurity[1,1])**2
        t_imp[n] = np.abs(1.0/MensembleStandingWaveImpurity[1,1])**2
        if r[n] < r_min:
            r_min = r[n]
            Delta3_for_r_min = Delta3
    handle1, = p.plot(Delta3Values, r, color='#000099', linestyle='-',
                      label=r'$|r_0|^2$', linewidth=common_line_width)
    handle2, = p.plot(Delta3Values, t, color='#009900', linestyle=':',
                      label=r'$|t_0|^2$', linewidth=common_line_width)
    handle3, = p.plot(Delta3Values, r_imp, color='#990000', linestyle='--',
                      label=r'$|r_1|^2$', linewidth=common_line_width)
    handle4, = p.plot(Delta3Values, t_imp, 'k-.',
                      label=r'$|t_1|^2$', linewidth=common_line_width)
    p.axvline(Delta3_for_r_min, color='k', linestyle=':')
    p.xlim(min_Delta3, max_Delta3)
    ax.annotate(r'$|r_0|^2$',
            xy=(Delta3Values[180], r[180]), xycoords='data',
            xytext=(0.03, 0.60), textcoords='data',
            size=10, va="center", ha="left",
            arrowprops=dict(arrowstyle="->",
                            shrinkA=0,
                            shrinkB=0,
                            connectionstyle="arc3,rad=0"),
    )
    ax.annotate(r'$|t_0|^2$',
            xy=(Delta3Values[160], t[160]), xycoords='data',
            xytext=(0.03, 0.30), textcoords='data',
            size=10, va="center", ha="left",
            arrowprops=dict(arrowstyle="->",
                            shrinkA=0,
                            shrinkB=0,
                            connectionstyle="arc3,rad=0"),
    )
    ax.annotate(r'$|r_1|^2$',
            xy=(Delta3Values[277], r_imp[277]), xycoords='data',
            xytext=(0.17, 0.85), textcoords='data',
            size=10, va="center", ha="left",
            arrowprops=dict(arrowstyle="->",
                            shrinkA=0,
                            shrinkB=0,
                            connectionstyle="arc3,rad=0"),
    )
    ax.annotate(r'$|t_1|^2$',
            xy=(Delta3Values[290], t_imp[290]), xycoords='data',
            xytext=(0.25, 0.30), textcoords='data',
            size=10, va="center", ha="left",
            arrowprops=dict(arrowstyle="->",
                            shrinkA=0,
                            shrinkB=0,
                            connectionstyle="arc3,rad=0"),
    )

def plot_with_specific_parameters():
    #Pick some incommensurate spacing between the atoms.
    #This will ensure that no atoms will be exactly on
    #the nodes of the standing wave
    kd = 0.5*np.pi
    #We treat Delta1, Deltap, g1d and gprime here as if they 
    #were scaled by the total decay rate 
    #\Gamma=\Gamma_{1D}+\Gamma'.
    #Thus Delta1 is actually Delta1/\Gamma etc.
    min_Delta3 = 0
    max_Delta3 = 0.7
    #Pick an odd number of points to make sure that we
    #hit Delta1=0 (where the reflection coefficient should
    #be equal to zero)
    num_Delta3 = 1001
    NAtoms = 10000
    g1d = 0.05
    #Note that \Gamma'/\Gamma=1-\Gamma_{1D}/\Gamma.
    gprime = 1-g1d
    gm = 0
    Deltap = -10
    Omega = 10
    plot_r_and_t(min_Delta3, max_Delta3, num_Delta3, NAtoms, kd, g1d,
                 gprime, gm, Deltap, Omega)
    p.xlabel(r'$\delta/\Gamma$')

usetex()
p.figure(figsize=(3.375,1.7))
ax = p.subplot(111)
plot_with_specific_parameters()
p.tight_layout(pad=0)
p.savefig('fig2.eps')
