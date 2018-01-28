/*
 * Copyright (c) 2017 Ivan Iakoupov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "bandstructure.h"
#include "ensemble_scattering.h"

#include "findroot.h"

#include <sstream>
#include <fstream>
#include <iostream>

namespace {

inline std::string levelSchemeToString(LevelScheme level_scheme) {
    switch (level_scheme) {
    case LevelScheme::dualV:
        return "dualV";
    case LevelScheme::dualColor:
        return "dualColor";
    case LevelScheme::LambdaType:
        return "lambda";
    default:
        assert(0 && "Unknown level scheme");
    }
}

struct DispersionRelationData
{
    double delta;
    std::vector<std::complex<double>> qd;
};

std::vector<DispersionRelationData>
generate_dispersion_relation_data(double delta_min, double delta_max,
                                  double g1d, int NAtoms, int NDeltad,
                                  double Deltac, double Deltad, double Omega_in,
                                  int classicalDrivePeriods, double shift,
                                  double distributionWidth, int randomSeed,
                                  LevelScheme level_scheme, int numPoints,
                                  bool logarithmicScale, bool regularPlacement)
{
    double Omega = Omega_in;
    if (level_scheme == LevelScheme::dualV) {
        // We divide the Omega by two to better match the
        // Lambda-type case. For dual-V, the Omega/2 as the
        // argument means that each of the two orthogonally
        // polarized running waves is of the form (Omega/2)*exp(\pm kz)
        // For Lambda-type, Omega means that the standing wave
        // of the classical drive is of the form Omega*cos(kz).
        // Hence, if the two running waves of the dual-V case
        // were not orthogonally polarized, they would have produced
        // the same standing wave as in the Lambda-type case.
        Omega /= 2;
    }
    QDOnBlochBandRandom qdBloch(classicalDrivePeriods, level_scheme,
                                regularPlacement);
    double delta_min_log;
    double delta_max_log;
    double delta_step;
    double delta_step_log;

    const int delta_num = numPoints;
    if (logarithmicScale) {
        delta_min_log = std::log10(delta_min);
        delta_max_log = std::log10(delta_max);
        delta_step_log = (delta_max_log - delta_min_log) / (delta_num - 1);
    } else {
        delta_step = (delta_max - delta_min) / (delta_num - 1);
    }
    std::vector<std::complex<double>> last_qd;
    std::vector<double> last_qd_re_diff;
    std::vector<double> qd_shifts;
    std::vector<DispersionRelationData> data(delta_num);
    const double branch_threshold = 0.1/NAtoms;
    const double branch_shift = 2*M_PI/NAtoms;
    for (int i = 0; i < delta_num; ++i) {
        double delta;
        if (logarithmicScale) {
            delta = std::pow(10, delta_min_log + i*delta_step_log);
        } else {
            delta = delta_min + i*delta_step;
        }
        qdBloch.setRandomSeed(randomSeed, NAtoms);
        const std::vector<std::complex<double>> qd
                = qdBloch.qd(NAtoms, NDeltad, delta, Deltac, Deltad, Omega, g1d, shift, distributionWidth);

        data[i].delta = delta;
        const int qd_size = qd.size();
        if (last_qd.empty()) {
            last_qd = std::vector<std::complex<double>>(qd_size);
        }
        if (last_qd_re_diff.empty()) {
            last_qd_re_diff = std::vector<double>(qd_size);
        }
        if (qd_shifts.empty()) {
            qd_shifts = std::vector<double>(qd_size, 0);
        }
        data[i].qd = std::vector<std::complex<double>>(qd_size);
        for(int j = 0; j < qd_size; ++j) {
            data[i].qd[j] = qd[j] + qd_shifts[j];
            if (!last_qd.empty() && !last_qd_re_diff.empty()) {
                double qd_re_diff = data[i].qd[j].real()-last_qd[j].real();
                if (std::abs(last_qd[j].real()-qd_shifts[j]-M_PI/NAtoms) < branch_threshold && qd_re_diff*last_qd_re_diff[j] < 0) {
                    data[i].qd[j] += branch_shift;
                    qd_re_diff += branch_shift;
                    qd_shifts[j] += branch_shift;
                    std::cout << "i = " << i
                              << ", j = " << j
                              << ", qd = " << qd[j]
                              << ", last_qd = " << last_qd[j]
                              << ", data[i].qd[j] = " << data[i].qd[j]
                              << std::endl;
                } else if (std::abs(last_qd[j].real()-qd_shifts[j]+M_PI/NAtoms) < branch_threshold && qd_re_diff*last_qd_re_diff[j] < 0) {
                    data[i].qd[j] -= branch_shift;
                    qd_re_diff -= branch_shift;
                    qd_shifts[j] -= branch_shift;
                    std::cout << "i = " << i
                              << ", j = " << j
                              << ", qd = " << qd[j]
                              << ", last_qd = " << last_qd[j]
                              << ", data[i].qd[j] = " << data[i].qd[j]
                              << std::endl;
                }
                last_qd_re_diff[j] = qd_re_diff;
                last_qd[j] = data[i].qd[j];
            }
        }
    }
    return data;
}

void write_dispersion_relation_data_to_file(
        const std::string &path, const std::vector<DispersionRelationData> &data)
{
    std::ofstream file(path);
    file.precision(17);
    file << "delta";

    // Assuming that the qd array has the same size
    // for all elements in data
    const int qd_size = data[0].qd.size();

    for (int i = 0; i < qd_size; ++i) {
        file << ';' << "qdRe" << i+1;
    }
    for (int i = 0; i < qd_size; ++i) {
        file << ';' << "qdIm" << i+1;
    }
    file << '\n';

    auto iter = data.cbegin();
    auto end = data.cend();
    for(; iter != end; ++iter) {
        file << iter->delta;
        for (int i = 0; i < qd_size; ++i) {
            file << ';' << iter->qd[i].real();
        }
        for (int i = 0; i < qd_size; ++i) {
            file << ';' << iter->qd[i].imag();
        }
        file << '\n';
    }
    std::cout << "Wrote to " << path << std::endl;
}

void generate_dispersion_relation_data_specific_params()
{
    const double delta_min = 1e-6;
    const double delta_max = 1;
    const bool logarithmicScale = true;
    const int numPoints = 1000;
    LevelScheme level_scheme = LevelScheme::LambdaType;
    //LevelScheme level_scheme = LevelScheme::dualV;
    //LevelScheme level_scheme = LevelScheme::dualColor;

    bool regularPlacement = false;

    const double g1d = 0.1;
    const int NAtoms = 10000;
    const int classicalDrivePeriods = NAtoms;
    const double Deltac = -90;
    const double Deltad = 1e-4;
    const double NDeltad = 10;
    const double Omega = 1;
    const int randomSeed = 12345;
    const double shift = 0;
    const double distributionWidth = 0;
    const std::vector<DispersionRelationData> data =
        generate_dispersion_relation_data(
            delta_min, delta_max, g1d, NAtoms, NDeltad, Deltac, Deltad, Omega,
            classicalDrivePeriods, shift, distributionWidth,
            randomSeed, level_scheme, numPoints, logarithmicScale,
            regularPlacement);
    std::stringstream fileNameStream;
    fileNameStream << "grating_dispersion_relation";
    fileNameStream << '_' << levelSchemeToString(level_scheme);
    if (regularPlacement) {
        fileNameStream << "_regular_placement";
    }
    fileNameStream << "_N_" << NAtoms
                   << "_g1d_" << g1d
                   << "_Deltac_" << Deltac;
    if (level_scheme == LevelScheme::dualColor) {
        fileNameStream << "_Deltad_" << Deltad
                       << "_NDeltad_" << NDeltad;
    }
    fileNameStream << "_Omega_" << Omega
                   << "_OmegaPeriods_" << classicalDrivePeriods
                   << "_seed_" << randomSeed
                   << ".txt";
    write_dispersion_relation_data_to_file(fileNameStream.str(), data);
}

struct TRData
{
    double delta;
    std::complex<double> t_regular_lambda;
    std::complex<double> r_regular_lambda;
    std::complex<double> t_random_lambda;
    std::complex<double> r_random_lambda;
    std::complex<double> t_regular_dualv;
    std::complex<double> r_regular_dualv;
    std::complex<double> t_random_dualv;
    std::complex<double> r_random_dualv;
};

void generate_t_r_data_specific_params()
{
    const double g1d = 0.1;
    const double min_delta = -0.02;
    const double max_delta = 0.02;
    const int num_delta = 10001;
    const double step_delta = (max_delta - min_delta) / (num_delta - 1);
    const double kd_ensemble = 0.5;
    const double Deltac = -90;
    const double Omega = 1;
    const int impurityShift = 0;
    const int NAtoms = 40000;
    const int randomSeed = 12345;
    int num_realizations = 100;

    int flagsRegular = 0;
    int flagsRandom = 0;
    flagsRandom |= ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS;
    EnsembleScattering ensemble_scattering_regular(flagsRegular);
    //Setting the random seed is only relevant for the randomly placed ensembles
    //ensemble_scattering_regular.setRandomSeed(randomSeed);
    std::vector<EnsembleScattering*> ensemble_scattering_random(num_realizations);
    for (int j = 0; j < num_realizations; ++j) {
        ensemble_scattering_random[j] = new EnsembleScattering(flagsRandom);
    }
    int flagsRegularDualV = ENSEMBLE_SCATTERING_DUAL_V_ATOMS;
    int flagsRandomDualV = ENSEMBLE_SCATTERING_DUAL_V_ATOMS;
    flagsRandomDualV |= ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS;
    EnsembleScattering ensemble_scattering_dualv_regular(
                flagsRegularDualV);
    //Setting the random seed is only relevant for the randomly placed ensembles
    //ensemble_scattering_dualv_regular.setRandomSeed(randomSeed);
    EnsembleScattering ensemble_scattering_dualv_random(
                flagsRandomDualV);
    ensemble_scattering_dualv_random.setRandomSeed(randomSeed);
    std::vector<TRData> data(num_delta);
    for (int i = 0; i < num_delta; ++i)
    {
        double delta = min_delta + i*step_delta;
        if (std::abs(delta) < 1e-16) {
            delta = 0;
        }

        const double kd = kd_ensemble;
        RandTCoefficients retRegular
                = ensemble_scattering_regular(delta, kd, kd,
                                                Deltac, g1d, Omega,
                                                NAtoms, NAtoms/2+impurityShift);
        std::complex<double> random_t(0,0);
        std::complex<double> random_r(0,0);
        for (int j = 0; j < num_realizations; ++j) {
            RandTCoefficients retRandom
                    = (*ensemble_scattering_random[j])(delta, kd, kd,
                                                         Deltac, g1d, Omega,
                                                         NAtoms, NAtoms/2+impurityShift);
            random_t += retRandom.t;
            random_r += retRandom.r;
        }
        random_t /= num_realizations;
        random_r /= num_realizations;
        RandTCoefficients retRegularDualV
                = ensemble_scattering_dualv_regular(delta, kd, kd,
                                                      Deltac, g1d, M_SQRT2*Omega/2,
                                                      NAtoms, NAtoms/2+impurityShift);
        RandTCoefficients retRandomDualV
                = ensemble_scattering_dualv_random(delta, kd, kd,
                                                     Deltac, g1d, M_SQRT2*Omega/2,
                                                     NAtoms, NAtoms/2+impurityShift);
        data[i].delta = delta;
        data[i].t_regular_lambda = retRegular.t;
        data[i].r_regular_lambda = retRegular.r;
        data[i].t_random_lambda = random_t;
        data[i].r_random_lambda = random_r;
        data[i].t_regular_dualv = retRegularDualV.t;
        data[i].r_regular_dualv = retRegularDualV.r;
        data[i].t_random_dualv = retRandomDualV.t;
        data[i].r_random_dualv = retRandomDualV.r;
    }
    for (int j = 0; j < num_realizations; ++j) {
        delete ensemble_scattering_random[j];
    }
    std::stringstream fileNameStream;
    fileNameStream << "grating_t_r";
    fileNameStream << "_N_" << NAtoms
                   << "_g1d_" << g1d
                   << "_Deltac_" << Deltac
                   << "_Omega_" << Omega
                   << "_kd_" << kd_ensemble
                   << "_seed_" << randomSeed
                   << ".txt";
    const std::string path = fileNameStream.str();
    std::ofstream file(path);
    file.precision(17);
    file << "delta" << ';'
         << "t_regular_lambda_re" << ';'
         << "t_regular_lambda_im" << ';'
         << "r_regular_lambda_re" << ';'
         << "r_regular_lambda_im" << ';'
         << "t_random_lambda_re" << ';'
         << "t_random_lambda_im" << ';'
         << "r_random_lambda_re" << ';'
         << "r_random_lambda_im" << ';'
         << "t_regular_dualv_re" << ';'
         << "t_regular_dualv_im" << ';'
         << "r_regular_dualv_re" << ';'
         << "r_regular_dualv_im" << ';'
         << "t_random_dualv_re" << ';'
         << "t_random_dualv_im" << ';'
         << "r_random_dualv_re" << ';'
         << "r_random_dualv_im" << '\n';

    for(const TRData d : data) {
        file << d.delta << ';'
             << d.t_regular_lambda.real() << ';'
             << d.t_regular_lambda.imag() << ';'
             << d.r_regular_lambda.real() << ';'
             << d.r_regular_lambda.imag() << ';'
             << d.t_random_lambda.real() << ';'
             << d.t_random_lambda.imag() << ';'
             << d.r_random_lambda.real() << ';'
             << d.r_random_lambda.imag() << ';'
             << d.t_regular_dualv.real() << ';'
             << d.t_regular_dualv.imag() << ';'
             << d.r_regular_dualv.real() << ';'
             << d.r_regular_dualv.imag() << ';'
             << d.t_random_dualv.real() << ';'
             << d.t_random_dualv.imag() << ';'
             << d.r_random_dualv.real() << ';'
             << d.r_random_dualv.imag() << '\n';
    }
    std::cout << "Wrote to " << path << std::endl;
}
} // unnamed namespace

int main(int argc, char *argv[])
{
    generate_dispersion_relation_data_specific_params();
    //generate_t_r_data_specific_params();
    return 0;
}
