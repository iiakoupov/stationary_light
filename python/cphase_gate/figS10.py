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

import matplotlib.pyplot as p
import numpy as np
from scipy.integrate import quad
from matplotlib_util import extract_params_from_file_name,\
                            read_column_names
from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

prefix = 'plot_data'

file_name = 'cphase_fidelity_NAtoms_aE_a_opt_lambda_sagnac_g1d_0.05_OmegaScattering_1_OmegaStorageRetrieval_1_kd_0.5.txt'

usetex()
p.figure(figsize=(3.3*2,2.5))
param_dict = extract_params_from_file_name(file_name)
full_path = os.path.join(prefix, file_name)
if not os.path.exists(full_path):
    print('Path {} doesn\'t exist'.format(full_path))
data = np.loadtxt(full_path, dtype=np.float64, delimiter=';',
                  unpack=True, skiprows=1)
column_names = read_column_names(full_path)
column_dic = dict(zip(column_names, range(len(column_names))))
NAtoms_index = column_dic['NAtoms']
g1d = param_dict['g1d']
gprime = 1-g1d
g1d_over_gprime = g1d/(1-g1d)
xdata = data[NAtoms_index]
sigma_analytical_data = np.sqrt(1.0/(xdata**(1.0/4)*np.pi**(3.0/2))*np.sqrt(gprime))
F_CJ_analytical_data1 = np.pi/(xdata**(1/2)*g1d_over_gprime)\
                        +np.pi**(3.0/2)*np.sqrt(1-g1d)/(g1d*xdata**(3.0/4))
F_CJ_analytical_data2 = 2*np.pi/(xdata**(1/2)*g1d_over_gprime)\
                        +np.pi**(3.0/2)*np.sqrt(1-g1d)/(g1d*xdata**(3.0/4))
F_CJ_cond_analytical_data1 = np.pi**2/(4*xdata*g1d_over_gprime**2)
F_CJ_cond_analytical_data2 = 11*np.pi**3*(g1d*(1-g1d)+(1-g1d)**2)/(16*xdata**(3/2)*g1d**2)


F_CJ_analytical_data3 = np.zeros_like(xdata)
for NAtoms_num, NAtoms_val in enumerate(xdata):
    Deltac_val = data[column_dic['Deltac']][NAtoms_num]
    sigma_val = data[column_dic['sigma']][NAtoms_num]
p.subplot(1,2,1)
p.loglog(xdata, 1-data[column_dic['F_CJ_tNoInteraction_one']], color='#009900', linestyle=':', label=r'$t_{\rm b}=1$ (numeric)', linewidth=common_line_width)
p.loglog(xdata, F_CJ_analytical_data1, 'k-.', label=r'$t_{\rm b}=1$ (analytic)', linewidth=common_line_width)
p.loglog(xdata, 1-data[column_dic['F_CJ']], color='#000099', linestyle='-', label=r'$t_{\rm b}<1$ (numeric)', linewidth=common_line_width)
p.loglog(xdata, F_CJ_analytical_data2, color='#990000', linestyle='--', label=r'$t_{\rm b}<1$ (analytic)', linewidth=common_line_width)
p.xlabel(r"$N$")
p.ylabel(r"$1-F_{\rm CJ}$")
p.title(r"(a)")
p.legend(loc='lower left')
p.ylim(1e-2,1)
p.subplot(1,2,2)
p.loglog(xdata, 1-data[column_dic['F_CJ_conditional_tNoInteraction_one']], color='#009900', linestyle=':', label=r'$t_{\rm b}=1$ (numeric)', linewidth=common_line_width)
p.loglog(xdata, F_CJ_cond_analytical_data1, 'k-.', label=r'$t_{\rm b}=1$ (analytic)', linewidth=common_line_width)
p.loglog(xdata, 1-data[column_dic['F_CJ_conditional']], color='#000099', linestyle='-', label=r'$t_{\rm b}<1$ (numeric)', linewidth=common_line_width)
p.loglog(xdata, F_CJ_cond_analytical_data2, color='#990000', linestyle='--', label=r'$t_{\rm b}<1$ (analytic)', linewidth=common_line_width)
p.xlabel(r"$N$")
p.ylabel(r"$1-F_{\rm CJ,cond}$")
p.title(r"(b)")
p.legend(loc='lower left')

p.tight_layout(pad=0.1)

p.savefig('figS10.eps')
