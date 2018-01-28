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
from scipy.optimize import minimize
from scipy.integrate import quad
from matplotlib_util import extract_params_from_file_name,\
                            read_column_names
from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

prefix = 'plot_data'

data_file_list = [
    'cphase_fidelity_kL1_aE_a_opt_lambda_sagnac_g1d_0.05_N_10000_OmegaScattering_1_OmegaStorageRetrieval_1_kd_0.5_kL2_0.txt',
    'cphase_fidelity_kL1_aE_a_opt_dualv_sym_sagnac_g1d_0.05_N_10000_OmegaScattering_1_OmegaStorageRetrieval_1_kd_0.5_kL2_0.txt',
]

usetex()
p.figure(figsize=(3.3*2,2.5))
data = []
column_dic = []
for file_name in data_file_list:
    param_dict = extract_params_from_file_name(file_name)
    full_path = os.path.join(prefix, file_name)
    if not os.path.exists(full_path):
        print('Path {} doesn\'t exist'.format(full_path))
        continue
    data.append(np.loadtxt(full_path, dtype=np.float64, delimiter=';',
                      unpack=True, skiprows=1))
    column_names = read_column_names(full_path)
    column_dic.append(dict(zip(column_names, range(len(column_names)))))
    g1d = param_dict['g1d']
    NAtoms = param_dict['N']

xdata_str = 'kL1'

# There is one value of kL1 that is very close
# to zero but not exactly so (because of round off
# errors presumably).
kL1_almost_zero_index = np.argmin(np.absolute(data[0][column_dic[0][xdata_str]]))

analytical_data0 = data[0][column_dic[0]['F_CJ_tNoInteraction_one']][kL1_almost_zero_index]\
                   -(0.5-5*(1-g1d)*np.pi/(8*g1d*np.sqrt(NAtoms)))*(np.pi*data[0][column_dic[0][xdata_str]])**2


p.subplot(1,2,1)
p.plot(data[0][column_dic[0][xdata_str]],
       data[0][column_dic[0]['F_CJ_conditional_tNoInteraction_one']],
       color='#000099', linestyle='-',
       label=r'$F_{\rm CJ,cond}$',
       linewidth=common_line_width)
p.plot(data[0][column_dic[0][xdata_str]],
       analytical_data0,
       color='#990000', linestyle='--',
       label=r'$F_{\rm CJ}$ (analytical)',
       linewidth=common_line_width)
p.plot(data[0][column_dic[0][xdata_str]],
       data[0][column_dic[0]['F_CJ_tNoInteraction_one']],
       color='#009900', linestyle=':',
       label=r'$F_{\rm CJ}$',
       linewidth=common_line_width)
p.xlabel(r"$k_0 l_1/\pi$")
p.ylim(0,1)
p.title(r"(a) $\Lambda$-type")
p.legend(loc='lower center', ncol=1)

p.subplot(1,2,2)
p.plot(data[1][column_dic[1][xdata_str]],
       data[1][column_dic[1]['F_CJ_conditional_tNoInteraction_one']],
       color='#000099', linestyle='-',
       label=r'$F_{\rm CJ,cond}$',
       linewidth=common_line_width)
p.plot(data[1][column_dic[1][xdata_str]],
       data[1][column_dic[1]['F_CJ_tNoInteraction_one']],
       color='#009900', linestyle=':',
       label=r'$F_{\rm CJ}$',
       linewidth=common_line_width)
p.xlabel(r"$k_0 l_1/\pi$")
p.ylim(0,1)
p.title(r"(b) dual-V")
p.legend(loc='lower center')
p.tight_layout(pad=0.1)

p.savefig('figS08.eps')
