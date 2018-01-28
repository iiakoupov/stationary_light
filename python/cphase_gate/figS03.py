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
    'cphase_fidelity_NAtoms_aE_a_opt_lambda_sagnac_g1d_0.05_OmegaScattering_1_OmegaStorageRetrieval_1_kd_0.5.txt',
]

usetex()
p.figure(figsize=(3.3*2,2.5))
for file_name in data_file_list:
    param_dict = extract_params_from_file_name(file_name)
    full_path = os.path.join(prefix, file_name)
    if not os.path.exists(full_path):
        print('Path {} doesn\'t exist'.format(full_path))
        continue
    data = np.loadtxt(full_path, dtype=np.float64, delimiter=';',
                      unpack=True, skiprows=1)
    column_names = read_column_names(full_path)
    column_dic = dict(zip(column_names, range(len(column_names))))
    NAtoms_index = column_dic['NAtoms']
    g1d = param_dict['g1d']
    g1d_over_gprime = g1d/(1-g1d)
    xdata = data[NAtoms_index]

    p.subplot(1,2,1)
    p.loglog(xdata, 1-data[column_dic['F_CJ_tNoInteraction_one']], color='#009900', linestyle=':', label=r'$1-F_{\rm CJ}$, $t_{\rm b}=1$', linewidth=common_line_width)
    p.loglog(xdata, 1-data[column_dic['P_success_tNoInteraction_one']], 'k-.', label=r'$1-P_{\rm suc}$, $t_{\rm b}=1$', linewidth=common_line_width)
    p.loglog(xdata, 1-data[column_dic['F_CJ']], color='#000099', linestyle='-', label=r'$1-F_{\rm CJ}$, $t_{\rm b}<1$', linewidth=common_line_width)
    p.loglog(xdata, 1-data[column_dic['P_success']], color='#990000', linestyle='--', label=r'$1-P_{\rm suc}$, $t_{\rm b}<1$', linewidth=common_line_width)
    p.xlabel(r"$N$")
    p.title(r"(a)")
    p.legend(loc='lower left')
    p.subplot(1,2,2)
    p.loglog(xdata, 1-data[column_dic['F_CJ_conditional_tNoInteraction_one']], color='#009900', linestyle=':', label=r'$1-F_{\rm CJ,cond}$, $t_{\rm b}=1$', linewidth=common_line_width)
    p.loglog(xdata, 1-data[column_dic['F_swap_tNoInteraction_one']], 'k-.', label=r'$1-F_{\rm swap}$, $t_{\rm b}=1$', linewidth=common_line_width)
    p.loglog(xdata, 1-data[column_dic['F_CJ_conditional']], color='#000099', linestyle='-', label=r'$1-F_{\rm CJ,cond}$, $t_{\rm b}<1$', linewidth=common_line_width)
    p.loglog(xdata, 1-data[column_dic['F_swap']], color='#990000', linestyle='--', label=r'$1-F_{\rm swap}$, $t_{\rm b}<1$', linewidth=common_line_width)
    p.xlabel(r"$N$")
    p.title(r"(b)")
    p.legend(loc='lower left')

p.tight_layout(pad=0.1)

p.savefig('figS03.eps')
