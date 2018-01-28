# Stationary light

This is the code for the publications:

Ivan Iakoupov, Johan R. Ott, Darrick E. Chang, Anders S. Sørensen  
"Dispersion relations for stationary light in one-dimensional atomic ensembles"  
[Phys. Rev. A 94, 053824 (2016)](http://link.aps.org/doi/10.1103/PhysRevA.94.053824)  
Preprint: [arXiv:1608.09016](https://arxiv.org/abs/1608.09016)

Ivan Iakoupov, Johannes Borregaard, Anders S. Sørensen  
"Controlled-phase Gate for Photons Based on Stationary Light"  
[Phys. Rev. Lett. 120, 010502 (2018)](https://link.aps.org/doi/10.1103/PhysRevLett.120.010502)  
Preprint: [arXiv:1610.09206](https://arxiv.org/abs/1610.09206)

The code has only ever been tested on Linux.

There are two subdirectories:
* "cpp"
* "python"

Subdirectory "cpp" contains C++ code for the time-consuming calculations.
To build the C++ code, run:

```sh
cd cpp
mkdir build
cd build
cmake ..
make
```

Two executables are produced: "dispersion_relation_plots" and "cphase_gate_plots" that correspond to the first and second article, respectively.

To generate the data for the plots using the executable "dispersion_relation_plots" one needs to change the hardcoded parameters in the file "dispersion_relation_plots.cpp" (functions "generate_dispersion_relation_data_specific_params()" and "generate_t_r_data_specific_params()"). This will produce the data files that are then used by the scripts in the "python" directory.

To to generate the data using the executable "cphase_gate_plots", in principle, one just need to run it. However, it will take EXTREMELY long time to complete (on the order of weeks or even months, depending on the hardware). Therefore, it may be better to look in the function "generate_article_plot_data()" in the file "cphase_gate_plots.cpp" and comment out most of the code, leaving only some parts. The comments indicate which figures the different parts of the code correspond to.

Subdirectory "python" contains Python scripts that produce the plots in the articles. The scripts are named "figN.py", where "N" is the number of the figure in the corresponding article. Some of the scripts do the less time-consuming calculations on the fly, but many of them use the data files generated by the above C++ code. Pre-generated data files are also included for convenience, so any Python script just needs to be run to produce a figure.