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
from matplotlib_util import extract_params_from_file_name,\
                            read_column_names
from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

def quadratic_dispersion_relation(N_u, Delta3, Deltac, Omega, g1d):
    gprime = 1-g1d
    m = -(N_u-1)*g1d**2/(N_u**2*(2*Deltac+1j*gprime)*Omega**2)
    qd = np.sqrt(2*m*Delta3)
    return qd

def dual_v_linear_dispersion_relation(Delta3, Deltac, Omega, g1d):
    gprime = 1-g1d
    Delta = Delta3+Deltac
    tilde_Delta = Delta+0.5j*gprime
    eta=Omega**2/(4*Delta3*tilde_Delta)
    return -g1d/(2*tilde_Delta)*(1-eta)/(1-2*eta)

prefix = './data_for_plots'

data_lambda_file_name = 'grating_dispersion_relation_lambda_N_10000_g1d_0.1_Deltac_-90_Omega_1_OmegaPeriods_5000_seed_12345.txt'
data_dualV_file_name = 'grating_dispersion_relation_dualV_N_10000_g1d_0.1_Deltac_-90_Omega_1_OmegaPeriods_5000_seed_12345.txt'

param_dict_dualV = extract_params_from_file_name(data_dualV_file_name)
full_path_dualV = os.path.join(prefix, data_dualV_file_name)
full_path_lambda = os.path.join(prefix, data_lambda_file_name)
data_dualV = p.loadtxt(full_path_dualV, dtype=p.float64, delimiter=';',
                 unpack=True, skiprows=1)
data_lambda = p.loadtxt(full_path_lambda, dtype=p.float64, delimiter=';',
                 unpack=True, skiprows=1)
column_names_lambda = read_column_names(full_path_lambda)
column_dic_lambda = dict(zip(column_names_lambda, range(len(column_names_lambda))))
column_names_dualV = read_column_names(full_path_dualV)
column_dic_dualV = dict(zip(column_names_dualV, range(len(column_names_dualV))))

g1d = 0.1
Deltac = -90
Omega = 1

usetex()
p.figure(figsize=(3.3,2.5))
ax = p.subplot(111)

delta_array_log = np.linspace(-6, 0, 500)
delta_array2 = 10**delta_array_log
delta_array2_len = len(delta_array2)
q_array_re_Lambda_cold_analytical = np.zeros_like(delta_array2)
q_array_im_Lambda_cold_analytical = np.zeros_like(delta_array2)

q_array_re_dualV = np.zeros_like(delta_array2)
q_array_im_dualV = np.zeros_like(delta_array2)
q_array_re_dualV_linear = np.zeros_like(delta_array2)
q_array_im_dualV_linear = np.zeros_like(delta_array2)
q_array_re_eit = np.zeros_like(delta_array2)
q_array_im_eit = np.zeros_like(delta_array2)
for n in range(delta_array2_len):
    q_n_Lambda_cold_analytical = pdrc.q_Lambda_cold(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_Lambda_cold_analytical[n] = q_n_Lambda_cold_analytical.real
    q_array_im_Lambda_cold_analytical[n] = q_n_Lambda_cold_analytical.imag

    q_n_dualV = pdrc.q_secular(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_dualV[n] = q_n_dualV.real
    q_array_im_dualV[n] = q_n_dualV.imag

    q_n_dualV_linear = dual_v_linear_dispersion_relation(delta_array2[n], Deltac, Omega, g1d)
    q_array_re_dualV_linear[n] = q_n_dualV_linear.real
    q_array_im_dualV_linear[n] = q_n_dualV_linear.imag

    q_n_eit = pdrc.q_eit(delta_array2[n], g1d, Deltac, Omega)
    q_array_re_eit[n] = q_n_eit.real
    q_array_im_eit[n] = q_n_eit.imag

usetex()

p.loglog(q_array_re_eit, delta_array2, 'k-',
         label='EIT',
         linewidth=common_line_width)
p.loglog(q_array_re_Lambda_cold_analytical, delta_array2, 'b-',
         label='$\Lambda$-type',
         linewidth=common_line_width)
p.loglog(np.absolute(q_array_re_dualV), delta_array2, 'r-',
         label='dual-V (secular)',
         linewidth=common_line_width)
p.loglog(data_lambda[column_dic_dualV['qdRe1']], data_lambda[0], 'm--',
         label=r'$\Lambda$-type numeric',
         linewidth=common_line_width)
p.loglog(q_array_re_dualV_linear, delta_array2, 'c-',
         label='dual-V (linear)',
         linewidth=common_line_width)
p.loglog(data_dualV[column_dic_dualV['qdRe1']], data_dualV[0], 'y--',
         label=r'dual-V numeric',
         linewidth=common_line_width)
p.loglog(data_dualV[column_dic_dualV['qdRe2']], data_dualV[0], 'g--',
         label=r'dual-V numeric',
         linewidth=common_line_width)

p.xlabel(r'${\rm Re}[q]/n_0$')
p.ylabel(r"$\delta/\Gamma$")
p.xlim(1e-6,1e-3)
p.tight_layout(pad=0.2)

p.savefig('fig3.eps')
