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

import pylab as p
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import random
from r_and_t_plot_common import ensemble_transfer_matrix_pi_half,\
                                impurity_unit_cell_pi_half

from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

def r_0(delta, NAtoms, kd, g1d, gprime, gm, Deltac, Omega):
    Delta = delta+Deltac
    MensembleStandingWave\
            = ensemble_transfer_matrix_pi_half(NAtoms, g1d, gprime, gm, Delta,
                                       Deltac, Omega)

    ret = -MensembleStandingWave[1,0]/MensembleStandingWave[1,1]
    return ret

def t_0(delta, NAtoms, kd, g1d, gprime, gm, Deltac, Omega):
    Delta = delta+Deltac
    MensembleStandingWave\
            = ensemble_transfer_matrix_pi_half(NAtoms, g1d, gprime, gm, Delta,
                                       Deltac, Omega)

    ret = 1.0/MensembleStandingWave[1,1]
    return ret

def delta_for_minimal_r_0(deltaStart, NAtoms, kd, g1d, gprime, gm, Deltac, Omega):
    res = minimize(lambda x: np.absolute(r_0(x[0],  NAtoms, kd, g1d, gprime, gm, Deltac, Omega)), deltaStart, method='Nelder-Mead')

    return res.x[0]

def plot_r_and_t_Omega():
    #Pick some incommensurate spacing between the atoms.
    #This will ensure that no atoms will be exactly on
    #the nodes of the standing wave
    kd = 0.5*np.pi
    #We treat Delta, Deltac, g1d and gprime here as if they 
    #were scaled by the total decay rate 
    #\Gamma=\Gamma_{1D}+\Gamma'.
    #Thus Delta is actually Delta/\Gamma etc.
    min_Omega = 1
    max_Omega = 100
    num_Omega = 1000
    NAtoms = 10000
    g1d = 0.05
    #Note that \Gamma'/\Gamma=1-\Gamma_{1D}/\Gamma.
    gprime = 1-g1d
    gm = 0
    Deltac = -10
    usetex()
    p.figure(figsize=(2*3.3,2.5))
    OmegaValues = np.linspace(min_Omega, max_Omega, num_Omega)
    r = np.zeros(num_Omega, dtype=np.float64)
    t = np.zeros(num_Omega, dtype=np.float64)
    r_imp = np.zeros(num_Omega, dtype=np.float64)
    t_imp = np.zeros(num_Omega, dtype=np.float64)
    #r_diff2 = np.zeros(num_Omega, dtype=np.complex128)
    t_diff2 = np.zeros(num_Omega, dtype=np.complex128)
    deltaStart = 0.0025
    for n, Omega in enumerate(OmegaValues):
        delta = delta_for_minimal_r_0(deltaStart, NAtoms, kd, g1d, gprime, gm, Deltac, Omega)
        deltaStart = delta
        Delta = delta+Deltac
        #r_0_func = lambda x: r_0(x, NAtoms, kd, g1d, gprime, gm, Deltac, Omega)
        #r_0_grad1 = grad(r_0_func)
        #r_0_grad2 = grad(r_0_grad1)
        t_0_func = lambda x: t_0(x, NAtoms, kd, g1d, gprime, gm, Deltac, Omega)
        t_0_grad1 = grad(t_0_func)
        t_0_grad2 = grad(t_0_grad1)
        MensembleStandingWave\
                = ensemble_transfer_matrix_pi_half(NAtoms, g1d, gprime, gm, Delta,
                                           Deltac, Omega)
        MHalfEnsembleStandingWave\
                = ensemble_transfer_matrix_pi_half(NAtoms/2, g1d, gprime, gm, Delta,
                                           Deltac, Omega)
        M_impurity_cell = impurity_unit_cell_pi_half(g1d, gprime, gm, Delta, Deltac, Omega)
        MensembleStandingWaveImpurity = MHalfEnsembleStandingWave*M_impurity_cell*MHalfEnsembleStandingWave
        r[n] = np.abs(-MensembleStandingWave[1,0]\
                      /MensembleStandingWave[1,1])**2
        t[n] = np.abs(1.0/MensembleStandingWave[1,1])**2
        r_imp[n] = np.abs(-MensembleStandingWaveImpurity[1,0]\
                          /MensembleStandingWaveImpurity[1,1])**2
        t_imp[n] = np.abs(1.0/MensembleStandingWaveImpurity[1,1])**2
        #r_diff2[n] = r_0_grad2(delta)
        t_diff2[n] = t_0_grad2(delta)
    t_analytical = (1-(g1d*(1-g1d)*NAtoms)/(16*Deltac**2)-(1-g1d)*OmegaValues**2*np.pi**2/(2*Deltac**2*g1d*NAtoms))**2
    r_imp_analytical = (1-4*np.pi**2*Deltac**2*(1-g1d)/(g1d**3*NAtoms**2)+32*np.pi**4*Deltac**2*(1-g1d)*OmegaValues**2/(g1d**5*NAtoms**4))**2
    w_analytical = 32*np.sqrt(2)*Deltac**2*OmegaValues**2*np.pi**2/(g1d**3*NAtoms**3)
    ax = p.subplot(121)
    #p.plot(OmegaValues, r, color='#000099', linestyle='-',
    #                  label=r'$|r_0|^2$', linewidth=common_line_width)
    p.plot(OmegaValues, t, color='#009900', linestyle=':',
                      label=r'$|t_0|^2$ (full)', linewidth=common_line_width)
    p.plot(OmegaValues, t_analytical-t_analytical[0]+t[0], color='k', linestyle='-.',
                      label=r'$|t_0|^2$ (approx)', linewidth=common_line_width)
    p.plot(OmegaValues, r_imp, color='#990000', linestyle='--',
                      label=r'$|r_1|^2$ (full)', linewidth=common_line_width)
    p.plot(OmegaValues, r_imp_analytical-r_imp_analytical[0]+r_imp[0], color='#000099', linestyle='-',
                      label=r'$|r_1|^2$ (approx)', linewidth=common_line_width)
    #p.plot(OmegaValues, t_imp, 'k-.',
    #                  label=r'$|t_1|^2$', linewidth=common_line_width)
    p.xlim(min_Omega, max_Omega)
    p.ylim(0.0, 1)
    p.xlabel(r'$\Omega_0/\Gamma$')
    p.title('(a)')
    p.legend(loc='lower right')
    ax = p.subplot(122)

    #r(\delta)=r(\delta_{res})+(2/w^2)(\delta-\delta_{res})^2
    #Hence (d/d\delta)r(\delta) at \delta=\delta_{res}
    #is equal to 4/w^2
    #Upon solving r_diff2=4/w^2, we get w=\sqrt{4/r_diff2}
    p.loglog(OmegaValues, np.real(np.sqrt(4.0/t_diff2)), color='#009900', linestyle=':',
                      label=r"full", linewidth=common_line_width)
    p.loglog(OmegaValues, w_analytical, color='k', linestyle='-.',
                      label=r"approximate", linewidth=common_line_width)
    p.ylabel(r'$w/\Gamma$')
    p.xlabel(r'$\Omega_0/\Gamma$')
    p.legend(loc='lower right')
    p.title('(b)')
    p.tight_layout(pad=0.2)

plot_r_and_t_Omega()
p.savefig('figS05.eps')
