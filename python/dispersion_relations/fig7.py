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
from matplotlib.ticker import MultipleLocator
from matplotlib_util import extract_params_from_file_name,\
                            read_column_names
import plot_dispersion_relations_common as pdrc
from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

prefix = 'data_for_plots'

file_name = 'grating_t_r_N_40000_g1d_0.1_Deltac_-90_Omega_1_kd_0.5_seed_12345.txt'



usetex()
p.figure(figsize=(3.3,4.5))
ax = p.subplot(211)
ax.xaxis.set_major_locator(MultipleLocator(base=0.001))

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
Omega = param_dict['Omega']
NAtoms = param_dict['N']

delta_array = np.linspace(-0.001, 0.002, 601)
delta_array_len = len(delta_array)
q_array_re_lambda_regular_N_u_2 = np.zeros_like(delta_array)
q_array_im_lambda_regular_N_u_2 = np.zeros_like(delta_array)
q_array_re_Lambda_cold_analytical = np.zeros_like(delta_array)
q_array_im_Lambda_cold_analytical = np.zeros_like(delta_array)

q_first_resonance = np.pi/NAtoms
q_second_resonance = 2*np.pi/NAtoms

delta_first_resonance_regular = 0
delta_second_resonance_regular = 0
delta_first_resonance_random = 0
delta_second_resonance_random = 0

for n in range(delta_array_len):
    q_n_Lambda_cold_analytical = pdrc.q_Lambda_cold(delta_array[n], g1d, Deltac, Omega)
    q_array_re_Lambda_cold_analytical[n] = q_n_Lambda_cold_analytical.real
    q_array_im_Lambda_cold_analytical[n] = q_n_Lambda_cold_analytical.imag
    if delta_array[n] > 0 and delta_first_resonance_random == 0 and q_array_re_Lambda_cold_analytical[n] > q_first_resonance:
        delta_first_resonance_random = delta_array[n]
    if delta_array[n] > 0 and delta_second_resonance_random == 0 and q_array_re_Lambda_cold_analytical[n] > q_second_resonance:
        delta_second_resonance_random = delta_array[n]
    q_n_Lambda_cold_1 = np.absolute(pdrc.q_eit_standing_wave(delta_array[n]+Deltac, Deltac, Omega, g1d, 2, 0))
    q_array_re_lambda_regular_N_u_2[n] = q_n_Lambda_cold_1.real
    q_array_im_lambda_regular_N_u_2[n] = q_n_Lambda_cold_1.imag
    if delta_array[n] > 0 and delta_first_resonance_regular == 0 and q_array_re_lambda_regular_N_u_2[n] > q_first_resonance:
        delta_first_resonance_regular = delta_array[n]
    if delta_array[n] > 0 and delta_second_resonance_regular == 0 and q_array_re_lambda_regular_N_u_2[n] > q_second_resonance:
        delta_second_resonance_regular = delta_array[n]


r_regular_lambda = data[column_dic['r_regular_lambda_re']] + 1j*data[column_dic['r_regular_lambda_im']]
r_random_lambda = data[column_dic['r_random_lambda_re']] + 1j*data[column_dic['r_random_lambda_im']]
r_regular_dualv = data[column_dic['r_regular_dualv_re']] + 1j*data[column_dic['r_regular_dualv_im']]
r_random_dualv = data[column_dic['r_random_dualv_re']] + 1j*data[column_dic['r_random_dualv_im']]
t_regular_lambda = data[column_dic['t_regular_lambda_re']] + 1j*data[column_dic['t_regular_lambda_im']]
t_random_lambda = data[column_dic['t_random_lambda_re']] + 1j*data[column_dic['t_random_lambda_im']]
t_regular_dualv = data[column_dic['t_regular_dualv_re']] + 1j*data[column_dic['t_regular_dualv_im']]
t_random_dualv = data[column_dic['t_random_dualv_re']] + 1j*data[column_dic['t_random_dualv_im']]


p.plot(data[delta_index], np.absolute(r_regular_lambda)**2, 'b-', label=r'$|r|$ regular lambda', linewidth=common_line_width)
p.plot(data[delta_index], np.absolute(r_random_lambda)**2, 'g--', label=r'$|r|$ random lambda', linewidth=common_line_width)
p.plot(data[delta_index], np.absolute(t_regular_lambda)**2, 'm:', label=r'$|t|$ regular lambda', linewidth=common_line_width)
p.plot(data[delta_index], np.absolute(t_random_lambda)**2, 'c-.', label=r'$|t|$ random lambda', linewidth=common_line_width)
p.plot(data[delta_index], np.absolute(r_random_dualv)**2, 'r--', label=r'$|r|$ random dualv', linewidth=common_line_width)
p.xlim(-0.001,0.002)
p.title(r'$(a)$')
p.xlabel(r'$\delta/\Gamma$')

ax2 = p.subplot(212)
ax2.xaxis.set_major_locator(MultipleLocator(base=0.001))
p.plot(delta_array, q_array_re_Lambda_cold_analytical, 'g--', linewidth=common_line_width)
p.plot(delta_array, q_array_re_lambda_regular_N_u_2, 'b-', linewidth=common_line_width)
p.xlim(-0.001,0.002)
p.title(r'$(b)$')
p.xlabel(r'$\delta/\Gamma$')
p.ylabel(r'${\rm Re}[q]/n_0$')
p.axhline(q_first_resonance, color='k', linestyle=':')
p.axhline(q_second_resonance, color='k', linestyle=':')
ax2.text(0, q_first_resonance, r'$\pi/N$', ha="center", va="bottom", size=10)
ax2.text(0, q_second_resonance, r'$2\pi/N$', ha="center", va="bottom", size=10)
p.axvline(delta_first_resonance_regular, color='k', linestyle=':')
p.axvline(delta_first_resonance_random, color='k', linestyle=':')
p.axvline(delta_second_resonance_regular, color='k', linestyle=':')
p.axvline(delta_second_resonance_random, color='k', linestyle=':')
p.tight_layout(pad=0.2)

p.savefig('fig7.eps')
