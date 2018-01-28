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
from matplotlib.ticker import NullLocator
from matplotlib_util import extract_params_from_file_name,\
                            read_column_names
from common_typeset_info import usetex, common_line_width

#Use matplotlib pre-2.0 version style
p.style.use('classic')

prefix = 'plot_data'

data_file_list = [
        'eit_storage_models_comparison_g1d_0.05_NAtoms_10000_OmegaStorageRetrieval_1_kd_0.266.txt',
        'eit_storage_models_comparison_g1d_0.05_NAtoms_10000_OmegaStorageRetrieval_1_kd_0.266_seed_12345.txt',
]

usetex()
title_list = [r'(a) regular placement', r'(b) random placement']
p.figure(figsize=(3.3*2,2.0))
for m, file_name in enumerate(data_file_list):
    ax = p.subplot(1, 2, m+1)
    ax.yaxis.set_major_locator(NullLocator())
    param_dict = extract_params_from_file_name(file_name)
    full_path = os.path.join(prefix, file_name)
    if not os.path.exists(full_path):
        print('Path {} doesn\'t exist'.format(full_path))
        continue
    data = p.loadtxt(full_path, dtype=p.float64, delimiter=';',
                     unpack=True, skiprows=1)
    column_names = read_column_names(full_path)
    column_dic = dict(zip(column_names, range(len(column_names))))

    g1d = param_dict['g1d']
    z_index = column_dic['z']
    S_dispersion_relation = data[column_dic['S_dispersion_relation_re']] + 1j*data[column_dic['S_dispersion_relation_im']]
    S_dispersion_relation_abs = np.absolute(S_dispersion_relation)
    S_kernel = data[column_dic['S_kernel_re']] + 1j*data[column_dic['S_kernel_im']]
    S_kernel_abs = np.absolute(S_kernel)
    S_discrete = data[column_dic['S_discrete_re']] + 1j*data[column_dic['S_discrete_im']]
    S_discrete_abs = np.absolute(S_discrete)

    p.plot(data[z_index], S_discrete_abs, color='#000099', linestyle='-', label=r'discrete', linewidth=common_line_width)
    p.plot(data[z_index], S_dispersion_relation_abs, color='#FF0000', linestyle='--', label=r'dispersion', linewidth=common_line_width)
    p.plot(data[z_index], S_kernel_abs, color='#00FF00', linestyle=':', label=r'kernel', linewidth=common_line_width)
    p.xlabel(r'$z/L$')
    p.title(title_list[m])
    p.ylim(0,0.025)
    p.ylabel('arb. units')
    p.legend(loc='upper right')

p.tight_layout(pad=0.1)

p.savefig('figS06.eps')
