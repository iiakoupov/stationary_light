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
from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

prefix = 'data_for_plots'

data_file_list = [
    'grating_t_r_N_40000_g1d_0.1_Deltac_-90_Omega_1_kd_0.5_seed_12345.txt'
]

usetex()
p.figure(figsize=(3.3,2.5))
ax = p.subplot(111)
ax.xaxis.set_major_locator(MultipleLocator(base=0.01))
for file_name in data_file_list:
    param_dict = extract_params_from_file_name(file_name)
    full_path = os.path.join(prefix, file_name)
    if not os.path.exists(full_path):
        print('Path {} doesn\'t exist'.format(full_path))
        continue
    data = p.loadtxt(full_path, dtype=p.float64, delimiter=';',
                     unpack=True, skiprows=1)
    column_names = read_column_names(full_path)
    column_dic = dict(zip(column_names, range(len(column_names))))
    delta_index = column_dic['delta']


    g1d = param_dict['g1d']
    Deltac = param_dict['Deltac']

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
    p.xlim(-0.02,0.02)
    p.xlabel(r'$\delta/\Gamma$')
    p.tight_layout(pad=0.2)

p.savefig('fig6.eps')
