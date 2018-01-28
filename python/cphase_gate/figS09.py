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
    'cphase_fidelity_OmegaScattering_and_OmegaStorageRetrieval_aE_a_opt_lambda_sagnac_g1d_0.05_N_10000_kd_0.5.txt',
    'cphase_fidelity_OmegaScattering_and_OmegaStorageRetrieval_aE_a_opt_dualv_sym_sagnac_g1d_0.05_N_10000_kd_0.266.txt',
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

xdata_str = 'OmegaScattering'

p.subplot(1,2,1)
p.plot(data[0][column_dic[0][xdata_str]],
       data[0][column_dic[0]['F_CJ_conditional_tNoInteraction_one']],
       color='#000099', linestyle='-',
       label=r'$F_{\rm CJ,cond}$, $\Lambda$-type',
       linewidth=common_line_width)
p.plot(data[1][column_dic[1][xdata_str]],
       data[1][column_dic[1]['F_CJ_conditional_tNoInteraction_one']],
       color='#990000', linestyle='--',
       label=r'$F_{\rm CJ,cond}$, dual-V',
       linewidth=common_line_width)
p.plot(data[0][column_dic[0][xdata_str]],
       data[0][column_dic[0]['F_CJ_tNoInteraction_one']],
       color='#009900', linestyle=':',
       label=r'$F_{\rm CJ}$, $\Lambda$-type',
       linewidth=common_line_width)
p.plot(data[1][column_dic[1][xdata_str]],
       data[1][column_dic[1]['F_CJ_tNoInteraction_one']],
       'k-.',
       label=r'$F_{\rm CJ}$, dual-V',
       linewidth=common_line_width)
p.xlabel(r"$\Omega_0/\Gamma$")
p.ylim(0.4,1)
p.title(r"(a)")
p.legend(loc='center left')

p.subplot(1,2,2)
p.plot(data[0][column_dic[0][xdata_str]],
       -data[0][column_dic[0]['Deltac']],
       color='#000099', linestyle='-',
       label=r'$-\Delta_{\rm c}/\Gamma$, $\Lambda$-type',
       linewidth=common_line_width)
p.plot(data[1][column_dic[1][xdata_str]],
       -data[1][column_dic[1]['Deltac']],
       color='#990000', linestyle='--',
       label=r'$-\Delta_{\rm c}/\Gamma$, dual-V',
       linewidth=common_line_width)
p.plot(data[0][column_dic[0][xdata_str]],
       data[0][column_dic[0]['delta']],
       color='#009900', linestyle=':',
       label=r'$\delta_{\rm res}/\Gamma$, $\Lambda$-type',
       linewidth=common_line_width)
p.plot(data[1][column_dic[1][xdata_str]],
       data[1][column_dic[1]['delta']],
       'k-.',
       label=r'$\delta_{\rm res}/\Gamma$, dual-V',
       linewidth=common_line_width)
p.xlabel(r"$\Omega_0/\Gamma$")
p.title(r"(b)")
p.legend(loc='upper left')
p.tight_layout(pad=0.1)

p.savefig('figS09.eps')
