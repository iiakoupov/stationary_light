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

# This plot is not present in the article, but was done to support the claim
# on the second paragraph of Sec. V "Scattering properties". There, we said that 
# 
# "
# In Figs. 6 and 7(a), we plot transmittance |t|^2 and reflectance |r|^2 for
# ensembles with regular (N_u = 2) and random (average of 100 ensemble
# realizations) placement of $\Lambda$-type atoms. The latter case can also be
# calculated using the continuum model together with Appendix C. The main
# visible difference between the discrete model with random placement and the
# continuum model is that the former has noise in the region
# $-0.01 \lesssim \delta/ \Gamma \leq 0$ due to finite number of ensemble
# realizations.
# "

import os.path

import pylab as p
import numpy as np
from matplotlib_util import extract_params_from_file_name,\
                            read_column_names

prefix = 'data_for_plots'

file_name = 'grating_t_r_N_40000_g1d_0.1_Deltac_-90_Omega_1_kd_0.5_seed_12345.txt'

def transfer_matrix_continuum_dualv(delta, g1d, NAtoms, Deltac, Omega):
    gprime = 1-g1d
    tDelta = delta+Deltac+0.5j*gprime
    deltaS = Omega**2/(2*tDelta)
    alpha1 = g1d/(2*tDelta)*(delta-deltaS/2)/(delta-deltaS)
    alpha2 = g1d/(2*tDelta)*(deltaS/2)/(delta-deltaS)
    #qOverN0 = np.sqrt(g1d**2/(2*tDelta)**2*delta/(delta-deltaS))
    qOverN0 = np.sqrt(alpha1**2-alpha2**2)
    M = np.zeros((2,2), dtype=np.complex128)
    M[0,0] = -alpha1
    M[0,1] = -alpha2
    M[1,0] = alpha2
    M[1,1] = alpha1
    if delta != 0:
        return np.cos(qOverN0*NAtoms)*np.identity(2)+1j*np.sin(qOverN0*NAtoms)/qOverN0*M
    else:
        return np.cos(qOverN0*NAtoms)*np.identity(2)+1j*NAtoms*M


def transfer_matrix_continuum_lambda(delta, g1d, NAtoms, Deltac, Omega):
    gprime = 1-g1d
    tDelta = delta+Deltac+0.5j*gprime
    deltaS = Omega**2/(2*tDelta)
    if delta == 0:
        alpha1 = 0
        alpha2 = 0
    else:
        alpha1 = g1d/(2*tDelta)/np.sqrt(1-2*deltaS/delta)
        alpha2 = g1d/(2*tDelta)*(-(2*np.sqrt(1-2*deltaS/delta)+2*deltaS/delta-2)/(2*deltaS/delta*np.sqrt(1-2*deltaS/delta)))
    qOverN0 = np.sqrt(alpha1**2-alpha2**2)
    M = np.zeros((2,2), dtype=np.complex128)
    M[0,0] = -alpha1
    M[0,1] = -alpha2
    M[1,0] = alpha2
    M[1,1] = alpha1
    if delta != 0:
        return np.cos(qOverN0*NAtoms)*np.identity(2)+1j*np.sin(qOverN0*NAtoms)/qOverN0*M
    else:
        return np.cos(qOverN0*NAtoms)*np.identity(2)+1j*NAtoms*M

p.figure(figsize=(6,4))
param_dict = extract_params_from_file_name(file_name)
full_path = os.path.join(prefix, file_name)
if not os.path.exists(full_path):
    print('Path {} doesn\'t exist'.format(full_path))
data = p.loadtxt(full_path, dtype=p.float64, delimiter=';',
                 unpack=True, skiprows=1)
column_names = read_column_names(full_path)
column_dic = dict(zip(column_names, range(len(column_names))))
delta_index = column_dic['delta']


g1d = param_dict['g1d']
Deltac = param_dict['Deltac']
NAtoms = param_dict['N']
Omega = param_dict['Omega']

r_continuum_dualv = np.zeros(len(data[delta_index]), dtype=np.complex128)
r_continuum_lambda = np.zeros(len(data[delta_index]), dtype=np.complex128)
for k, delta in enumerate(data[delta_index]):
    M_dualv = transfer_matrix_continuum_dualv(delta, g1d, NAtoms, Deltac, Omega*np.sqrt(2))
    M_lambda = transfer_matrix_continuum_lambda(delta, g1d, NAtoms, Deltac, Omega)
    r_continuum_dualv[k] = -M_dualv[1,0]/M_dualv[1,1]
    r_continuum_lambda[k] = -M_lambda[1,0]/M_lambda[1,1]

r_regular_lambda = data[column_dic['r_regular_lambda_re']] + 1j*data[column_dic['r_regular_lambda_im']]
r_random_lambda = data[column_dic['r_random_lambda_re']] + 1j*data[column_dic['r_random_lambda_im']]
r_regular_dualv = data[column_dic['r_regular_dualv_re']] + 1j*data[column_dic['r_regular_dualv_im']]
r_random_dualv = data[column_dic['r_random_dualv_re']] + 1j*data[column_dic['r_random_dualv_im']]
p.plot(data[delta_index], np.absolute(r_regular_lambda)**2, label=r'|r|^2 regular lambda')
p.plot(data[delta_index], np.absolute(r_random_lambda)**2, label=r'|r|^2 random lambda')
p.plot(data[delta_index], np.absolute(r_continuum_lambda)**2, linestyle='--', label=r'|r|^2 continuum lambda')
p.plot(data[delta_index], np.absolute(r_continuum_dualv)**2, linestyle='--', label=r'|r|^2 continuum dualv')
#p.xlim(-0.001,0.001)
p.xlabel(r'$\delta$')
p.legend(loc='upper right')
p.tight_layout()

p.savefig('extra_t_r_plot_continuum_model.eps')
