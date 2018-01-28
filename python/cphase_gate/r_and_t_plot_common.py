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
import random

def two_level_minus_r_over_t(g1d, gprime, Delta1):
    """
    Minus the reflection coefficient of a two-level atom divided
    by the transmission coefficient. The transfer matrix for the
    two-level atoms will be written in terms of this ratio.
    """
    #Assume real Omega
    return g1d/(gprime-2j*Delta1)

def Lambda_type_minus_r_over_t(g1d, gprime, gm, Delta1, Deltap, Omega):
    """
    Minus the reflection coefficient of a Lambda-type atom divided
    by the transmission coefficient. The transfer matrix for the
    Lambda-type atoms will be written in terms of this ratio.
    """
    #Assume real Omega
    return g1d*(Delta1-Deltap+0.5j*gm)\
           /((gprime-2j*Delta1)*(Delta1-Deltap+0.5j*gm)+2j*Omega**2)

def ensemble_transfer_matrix(NAtoms, kd, g1d, gprime, gm, Delta1, Deltap,
                             Omega):
    """
    NAtoms: The number of atoms
    kd:     The product of the wavevector k of the input quantum field
            (and also approximately the wavevector of the classical drive)
            and the distance between the atoms d. For example kd=pi means
            that atoms are situated half a wavelength apart.
    g1d:    \Gamma_{1D}, the decay rate into the guided mode
    gprime: \Gamma', the decay rate into all the other modes
    gm:     \Gamma_m, the decay rate from the meta-stable state
    Delta1: The detuning of the carrier frequency \omega_L of the
            quantum field from the atomic transition frequency \omega_{ab}.
            Delta1 = \omega_L - \omega_{ab}
    Deltap: The detuning of the classical drive frequency \omega_p from
            the transition bc. Deltap = \omega_p - \omega_{bc}
    """
    Mf = np.mat([[np.exp(1j*kd), 0], [0, np.exp(-1j*kd)]])

    MensembleStandingWave = np.mat([[1,0],[0,1]])

    xiConstant = Lambda_type_minus_r_over_t(g1d, gprime, gm, Delta1,
                                            Deltap, Omega)
    for n in range(NAtoms):
        #This corresponds the the classical drive having the same
        #frequency as the quantum transition. Thus the intensity
        #of the classical drive oscillates with lambda/2
        OmegaStandingWave = Omega*np.cos(kd*n)
        xiStandingWave = Lambda_type_minus_r_over_t(g1d, gprime, gm, Delta1,
                                                    Deltap, OmegaStandingWave)
        #Fill in the scattering matrices for the current atom
        MatomStandingWave = np.mat([[1-xiStandingWave, -xiStandingWave],
                                    [xiStandingWave, 1+xiStandingWave]])
        #Multiply the scattering and free propagation matrices
        #onto the ensemble matrices
        MensembleStandingWave = Mf*MatomStandingWave*MensembleStandingWave
    return MensembleStandingWave

def ensemble_transfer_matrix_pi_half(NAtoms, g1d, gprime, gm, Delta1, Deltap,
                                     Omega):
    """
    NAtoms: The number of atoms
    g1d:    \Gamma_{1D}, the decay rate into the guided mode
    gprime: \Gamma', the decay rate into all the other modes
    gm:     \Gamma_m, the decay rate from the meta-stable state
    Delta1: The detuning of the carrier frequency \omega_L of the
            quantum field from the atomic transition frequency \omega_{ab}.
            Delta1 = \omega_L - \omega_{ab}
    Deltap: The detuning of the classical drive frequency \omega_p from
            the transition bc. Deltap = \omega_p - \omega_{bc}
    """
    kd = 0.5*np.pi

    Mf = np.mat([[np.exp(1j*kd), 0], [0, np.exp(-1j*kd)]])

    beta3 = Lambda_type_minus_r_over_t(g1d, gprime, gm, Delta1,
                                       Deltap, Omega)
    M3 = np.mat([[1-beta3, -beta3], [beta3, 1+beta3]])

    beta2 = two_level_minus_r_over_t(g1d, gprime, Delta1)
    M2 = np.mat([[1-beta2, -beta2], [beta2, 1+beta2]])

    Mcell = Mf*M2*Mf*M3
    NCells = NAtoms/2

    #diag, V = np.linalg.eig(Mcell)
    #DN = np.diag(diag**(NCells))
    #V_inv = np.linalg.inv(V)

    #return V*DN*V_inv

    theta = np.arccos(0.5*np.trace(Mcell))
    Id = np.mat([[1,0],[0,1]])

    Mensemble\
            = np.mat([[np.cos(NCells*theta)+1j*1j*np.sin(NCells*theta)/np.sin(theta)*(Mcell[1,1]-Mcell[0,0])/2, 
                 -1j*1j*np.sin(NCells*theta)/np.sin(theta)*Mcell[0,1]],
                [-1j*1j*np.sin(NCells*theta)/np.sin(theta)*Mcell[1,0],
                 np.cos(NCells*theta)-1j*1j*np.sin(NCells*theta)/np.sin(theta)*(Mcell[1,1]-Mcell[0,0])/2]])
    return Mensemble

def impurity_unit_cell_pi_half(g1d, gprime, gm, Delta1, Deltap, Omega):
    """
    NAtoms: The number of atoms
    g1d:    \Gamma_{1D}, the decay rate into the guided mode
    gprime: \Gamma', the decay rate into all the other modes
    gm:     \Gamma_m, the decay rate from the meta-stable state
    Delta1: The detuning of the carrier frequency \omega_L of the
            quantum field from the atomic transition frequency \omega_{ab}.
            Delta1 = \omega_L - \omega_{ab}
    Deltap: The detuning of the classical drive frequency \omega_p from
            the transition bc. Deltap = \omega_p - \omega_{bc}
    """
    kd = 0.5*np.pi

    Mf = np.mat([[np.exp(1j*kd), 0], [0, np.exp(-1j*kd)]])

    beta2_imp = two_level_minus_r_over_t(g1d, gprime, 0)
    M2_imp = np.mat([[1-beta2_imp, -beta2_imp], [beta2_imp, 1+beta2_imp]])

    beta2 = two_level_minus_r_over_t(g1d, gprime, Delta1)
    M2 = np.mat([[1-beta2, -beta2], [beta2, 1+beta2]])

    M_impurity_cell = Mf*M2*Mf*M2_imp
    return M_impurity_cell

