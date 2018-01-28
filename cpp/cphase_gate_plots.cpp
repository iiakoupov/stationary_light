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

#include "cphase_gate.h"
#include "ensemble_scattering.h"

#include "urandom.h"
#include "random_seeds.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <vector>
#include <fstream>
#include <iostream>
#include "time.h"

#include <stdio.h>
#include <ftw.h>
#include <unistd.h>

namespace {
struct CphaseGateFidelityData
{
    double P_success;
    double F_swap;
    double F_CJ;
    double F_CJ_conditional;
    double P_success_tNoInteraction_one;
    double F_swap_tNoInteraction_one;
    double F_CJ_tNoInteraction_one;
    double F_CJ_conditional_tNoInteraction_one;
    double single_photon_storage_retrieval_eff;
    double kd_ensemble;
    double kL1;
    double kL2;
    double g1d;
    int NAtoms;
    double delta;
    double Deltac;
    double OmegaScattering;
    double OmegaStorageRetrieval;
    double sigma;
    double tNoInteractionAbs;
    double tNoInteractionArg;
    double t_storage;
    double t_retrieval;
    double t_to_pass_ensemble;
};

enum class WhatToVary {
    varyNAtoms,
    varyG1d,
    varyOmegaScattering,
    varyOmegaStorageRetrieval,
    varyOmegaScatteringAndStorageRetrievalEqually,
    varyKD,
    varyKL1,
    varyKL2,
    varySigma,
    varySigmaAndDeltac
};

inline std::string whatToVaryToString(WhatToVary whatToVary) {
    switch (whatToVary) {
    case WhatToVary::varyNAtoms:
        return "NAtoms";
    case WhatToVary::varyG1d:
        return "g1d";
    case WhatToVary::varyOmegaScattering:
        return "OmegaScattering";
    case WhatToVary::varyOmegaStorageRetrieval:
        return "OmegaStorageRetrieval";
    case WhatToVary::varyKD:
        return "kd";
    case WhatToVary::varyKL1:
        return "kL1";
    case WhatToVary::varyKL2:
        return "kL2";
    default:
        assert(0 && "Unknown or composite quantity to vary!");
    }
}

inline std::string whatToVaryToFileName(WhatToVary whatToVary) {
    if (whatToVary == WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually) {
        std::stringstream ret;
        ret << whatToVaryToString(WhatToVary::varyOmegaScattering)
            << "_and_"
            << whatToVaryToString(WhatToVary::varyOmegaStorageRetrieval);
        return ret.str();
    } else if (whatToVary == WhatToVary::varySigmaAndDeltac) {
        std::stringstream ret;
        ret << whatToVaryToString(WhatToVary::varySigma)
            << "_and_"
            << whatToVaryToString(WhatToVary::varyOmegaStorageRetrieval);
        return ret.str();
    } else {
        return whatToVaryToString(whatToVary);
    }
}

inline std::string whatToVaryToTableHeader(WhatToVary whatToVary) {
    if (whatToVary == WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually) {
        std::stringstream ret;
        ret << whatToVaryToString(WhatToVary::varyOmegaScattering)
            << ';'
            << whatToVaryToString(WhatToVary::varyOmegaStorageRetrieval);
        return ret.str();
    } else if (whatToVary == WhatToVary::varySigma || whatToVary == WhatToVary::varySigmaAndDeltac) {
        // The assumption is that these values
        // are already added anyway.
        return "";
    } else {
        return whatToVaryToString(whatToVary);
    }
}

template <typename Stream>
inline void appendWhatToVaryDataPointValueToSteam(Stream &file, const CphaseGateFidelityData &dataPoint, WhatToVary whatToVary)
{
    switch (whatToVary) {
    case WhatToVary::varyNAtoms:
        file << dataPoint.NAtoms;
        break;
    case WhatToVary::varyG1d:
        file << dataPoint.g1d;
        break;
    case WhatToVary::varyOmegaScattering:
        file << dataPoint.OmegaScattering;
        break;
    case WhatToVary::varyOmegaStorageRetrieval:
        file << dataPoint.OmegaStorageRetrieval;
        break;
    case WhatToVary::varyKD:
        file << dataPoint.kd_ensemble;
        break;
    case WhatToVary::varyKL1:
        file << dataPoint.kL1;
        break;
    case WhatToVary::varyKL2:
        file << dataPoint.kL2;
        break;
    case WhatToVary::varySigma:
        file << dataPoint.sigma;
    default:
        assert(0 && "Unknown or composite quantity to vary!");
    }
}

template <typename Stream>
inline void appendWhatToVaryDataValuesToTableRow(Stream &file, const CphaseGateFidelityData &dataPoint, WhatToVary whatToVary)
{
    if (whatToVary == WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually) {
        file << dataPoint.OmegaScattering << ';'
             << dataPoint.OmegaStorageRetrieval << ';';
    } else if (whatToVary == WhatToVary::varySigma || whatToVary == WhatToVary::varySigmaAndDeltac) {
        // The assumption is that these values
        // are already added anyway.
        return;
    } else {
        appendWhatToVaryDataPointValueToSteam(file, dataPoint, whatToVary);
        file << ';';
    }
}

template <typename Stream>
inline void appendWhatToVaryDataValuesToFileName(Stream &fileNameStream, const CphaseGateFidelityData &dataPoint, WhatToVary whatToVary)
{
    if (whatToVary == WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually) {
        fileNameStream << "OmegaScattering_"
                       << dataPoint.OmegaScattering
                       << "_OmegaStorageRetrieval_"
                       << dataPoint.OmegaStorageRetrieval;
    } else if (whatToVary == WhatToVary::varySigmaAndDeltac) {
        fileNameStream << "sigma_"
                       << dataPoint.sigma
                       << "_Deltac_"
                       << dataPoint.Deltac;
        return;
    } else {
        fileNameStream << whatToVaryToString(whatToVary) << '_';
        appendWhatToVaryDataPointValueToSteam(fileNameStream, dataPoint, whatToVary);
    }
}

int unlink_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
{
    int rv = remove(fpath);

    if (rv)
        perror(fpath);

    return rv;
}

int rmrf(const char *path)
{
    return nftw(path, unlink_cb, 64, FTW_DEPTH | FTW_PHYS);
}

void write_cphase_gate_fidelity_data_to_file(
        const std::string &path,
        const std::vector<CphaseGateFidelityData> &data,
        WhatToVary whatToVary,
        bool appendOmegaValues)
{
    std::ofstream file(path);
    // If an IEEE 754 double precision is converted to a
    // decimal string with at least 17 significant digits
    // and then converted back to double, then the final
    // number must match the original.
    file.precision(17);

    std::string whatToVaryHeaderStr = whatToVaryToTableHeader(whatToVary);
    if (!whatToVaryHeaderStr.empty()) {
        // The empty check prevents putting a ";"
        // in the beginning.
        file << whatToVaryHeaderStr << ';';
    }
    if (appendOmegaValues) {
        file << "OmegaScattering" << ';'
             << "OmegaStorageRetrieval" << ';';
    }
    file << "sigma" << ';'
         << "Deltac" << ';'
         << "delta" << ';'
         << "tNoInteractionAbs" << ';'
         << "tNoInteractionArg" << ';'
         << "t_storage" << ';'
         << "t_retrieval" << ';'
         << "t_to_pass_ensemble" << ';'
         << "P_success" << ';'
         << "F_swap" << ';'
         << "F_CJ" << ';'
         << "F_CJ_conditional" << ';'
         << "P_success_tNoInteraction_one" << ';'
         << "F_swap_tNoInteraction_one" << ';'
         << "F_CJ_tNoInteraction_one" << ';'
         << "F_CJ_conditional_tNoInteraction_one" << ';'
         << "single_photon_storage_retrieval_eff" << '\n';
    for (const CphaseGateFidelityData &dataPoint : data) {
        appendWhatToVaryDataValuesToTableRow(file, dataPoint, whatToVary);
        if (appendOmegaValues) {
            file << dataPoint.OmegaScattering << ';'
                 << dataPoint.OmegaStorageRetrieval << ';';
        }
        file << dataPoint.sigma << ';'
             << dataPoint.Deltac << ';'
             << dataPoint.delta << ';'
             << dataPoint.tNoInteractionAbs << ';'
             << dataPoint.tNoInteractionArg << ';'
             << dataPoint.t_storage << ';'
             << dataPoint.t_retrieval << ';'
             << dataPoint.t_to_pass_ensemble << ';'
             << dataPoint.P_success << ';'
             << dataPoint.F_swap << ';'
             << dataPoint.F_CJ << ';'
             << dataPoint.F_CJ_conditional << ';'
             << dataPoint.P_success_tNoInteraction_one << ';'
             << dataPoint.F_swap_tNoInteraction_one << ';'
             << dataPoint.F_CJ_tNoInteraction_one << ';'
             << dataPoint.F_CJ_conditional_tNoInteraction_one << ';'
             << dataPoint.single_photon_storage_retrieval_eff << '\n';
    }
    std::cout << "Wrote to: " << path << std::endl;
}

struct ExtraTNoInteractionFidPoints
{
    std::vector<std::complex<double>> tNoInteraction;
    std::vector<CphaseFidelities> fid;
};

void write_cphase_gate_fidelity_extra_tNoInteraction_data_to_file(
        const std::string &path,
        const std::vector<CphaseGateFidelityData> &data,
        const std::vector<ExtraTNoInteractionFidPoints> &extraFidData,
        WhatToVary whatToVary,
        bool appendOmegaValues)
{
    const int dataSize = data.size();
    const int extraDataRowSize = extraFidData[0].fid.size();
    std::ofstream file(path);
    // If an IEEE 754 double precision is converted to a
    // decimal string with at least 17 significant digits
    // and then converted back to double, then the final
    // number must match the original.
    file.precision(17);
    std::string whatToVaryHeaderStr = whatToVaryToTableHeader(whatToVary);
    if (!whatToVaryHeaderStr.empty()) {
        // The empty check prevents putting a ";"
        // in the beginning.
        file << whatToVaryHeaderStr << ';';
    }
    if (appendOmegaValues) {
        file << "OmegaScattering" << ';'
             << "OmegaStorageRetrieval" << ';';
    }
    file << "sigma" << ';'
         << "Deltac" << ';'
         << "delta" << ';';
    for (int j = 0; j < extraDataRowSize; ++j) {
         file << "tNoInteractionAbs" << j << ';'
              << "tNoInteractionArg" << j << ';';
    }
    file << "t_storage" << ';'
         << "t_retrieval" << ';'
         << "t_to_pass_ensemble" << ';';
    for (int j = 0; j < extraDataRowSize; ++j) {
         file << "P_success" << j << ';'
              << "F_swap" << j << ';'
              << "F_CJ" << j << ';'
              << "F_CJ_conditional" << j << ';';
    }
    file << "single_photon_storage_retrieval_eff" << '\n';
    for (int i = 0; i < dataSize; ++i) {
        appendWhatToVaryDataValuesToTableRow(file, data[i], whatToVary);
        if (appendOmegaValues) {
            file << data[i].OmegaScattering << ';'
                 << data[i].OmegaStorageRetrieval << ';';
        }
        file << data[i].sigma << ';'
             << data[i].Deltac << ';'
             << data[i].delta << ';';
        for (int j = 0; j < extraDataRowSize; ++j) {
            file << extraFidData[i].tNoInteraction[j].real() << ';'
                 << extraFidData[i].tNoInteraction[j].imag() << ';';
        }
        file << data[i].t_storage << ';'
             << data[i].t_retrieval << ';'
             << data[i].t_to_pass_ensemble << ';';
        for (int j = 0; j < extraDataRowSize; ++j) {
            file << extraFidData[i].fid[j].P_success << ';'
                 << extraFidData[i].fid[j].F_swap << ';'
                 << extraFidData[i].fid[j].F_CJ << ';'
                 << extraFidData[i].fid[j].F_CJ_conditional << ';';
        }
        file << data[i].single_photon_storage_retrieval_eff << '\n';
    }
    std::cout << "Wrote to: " << path << std::endl;
}

void write_cphase_gate_fidelity_seed_specific_data_to_file(
        const std::string &nameStreamDumpStr,
        const std::vector<unsigned long int> randomSeeds,
        const std::vector<CphaseGateFidelityData> &data,
        const std::vector<std::vector<CphaseDiagnosticData>> &dataPerSeed,
        const std::vector<SpinwaveAndFieldVector> &solData,
        EnsembleScattering &ensemble_scattering,
        int cphaseGateFlags,
        bool dumpSpinwavesAndFields,
        WhatToVary whatToVary)
{
    const int dataSize = data.size();
    int numRandomSeeds = randomSeeds.size();
    if (!(cphaseGateFlags & CPHASE_GATE_RANDOM_ATOM_POSITIONS)) {
        numRandomSeeds = 1;
    }
    assert(dataPerSeed.size() == dataSize && "dataPerSeed.size() != data.size()");
    #pragma omp parallel for
    for (int n = 0; n < numRandomSeeds; ++n) {
        std::stringstream nameStreamDumpSeed;
        if (cphaseGateFlags & CPHASE_GATE_RANDOM_ATOM_POSITIONS) {
            nameStreamDumpSeed << nameStreamDumpStr << '/' << "seed_" << randomSeeds[n];
            int err = mkdir(nameStreamDumpSeed.str().c_str(), 0777);
            bool dumpDataDirExists = true;
            if (err == -1 && errno != EEXIST) {
                std::cout << "Failed making dump data directory "
                          << nameStreamDumpSeed.str() << ": "
                          << strerror(errno) << std::endl;
                dumpDataDirExists = false;
                continue;
            }
        } else {
            nameStreamDumpSeed << nameStreamDumpStr;
        }
        std::stringstream pathStream;
        pathStream << nameStreamDumpSeed.str() << '/'
                   << "additional_data.txt";
        std::ofstream file(pathStream.str());
        // If an IEEE 754 double precision is converted to a
        // decimal string with at least 17 significant digits
        // and then converted back to double, then the final
        // number must match the original.
        file.precision(17);
        std::string whatToVaryHeaderStr = whatToVaryToTableHeader(whatToVary);
        if (!whatToVaryHeaderStr.empty()) {
            // The empty check prevents putting a ";"
            // in the beginning.
            file << whatToVaryHeaderStr << ';';
        }
        file << "E_without_scattering_norm" << ';'
             << "E_with_scattering_norm" << ';'
             << "spinwave_after_storage_norm" << ';'
             << "spinwave_after_scattering_norm" << ';'
             << "impurityRfromE_Re" << ';'
             << "impurityRfromE_Im" << ';'
             << "impurityRfromSpinwave_Re" << ';'
             << "impurityRfromSpinwave_Im" << ';'
             << "noImpurityR_Re" << ';'
             << "noImpurityR_Im" << '\n';
        for (int i = 0; i < dataSize; ++i) {
            appendWhatToVaryDataValuesToTableRow(file, data[i], whatToVary);
            file << dataPerSeed[i][n].E_without_scattering_norm << ';'
                 << dataPerSeed[i][n].E_with_scattering_norm << ';'
                 << dataPerSeed[i][n].spinwave_after_storage_norm << ';'
                 << dataPerSeed[i][n].spinwave_after_scattering_norm << ';'
                 << dataPerSeed[i][n].impurityRfromE.real() << ';'
                 << dataPerSeed[i][n].impurityRfromE.imag() << ';'
                 << dataPerSeed[i][n].impurityRfromSpinwave.real() << ';'
                 << dataPerSeed[i][n].impurityRfromSpinwave.imag() << ';'
                 << dataPerSeed[i][n].noImpurityR.real() << ';'
                 << dataPerSeed[i][n].noImpurityR.imag() << '\n';
        }

        if (!dumpSpinwavesAndFields) {
            continue;
        }

        ensemble_scattering.setRandomSeed(randomSeeds[n]);
        for (int j = 0; j < dataSize; ++j) {
            const int NAtoms = data[j].NAtoms;
            const double g1d = data[j].g1d;
            const double kd_ensemble = data[j].kd_ensemble;
            const double kL1 = data[j].kL1;
            const double kL2 = data[j].kL2;
            std::stringstream nameStreamDumpFile;
            nameStreamDumpFile << nameStreamDumpSeed.str() << '/'
                               << "spinwaves_";
            appendWhatToVaryDataValuesToFileName(nameStreamDumpFile, data[j], whatToVary);

            nameStreamDumpFile << ".txt";
            std::ofstream fileFull(nameStreamDumpFile.str());
            fileFull.unsetf(std::ios::floatfield);
            fileFull.precision(17);
            fileFull
            << "z_without_scattering" << ";"
            << "z_with_scattering" << ";"
            << "psi_after_storage0_Re" << ";"
            << "psi_after_storage0_Im" << ";"
            << "psi_after_storage1_Re" << ";"
            << "psi_after_storage1_Im" << ";"
            << "psi_after_scattering_Re" << ";"
            << "psi_after_scattering_Im" << ";"
            << "r_impurity_Re" << ";"
            << "r_impurity_Im" << ";"
            << "t_impurity_Re" << ";"
            << "t_impurity_Im" << ";"
            << "R_n_Re" << ";"
            << "R_n_Im" << '\n';
            for (int i = 0; i < NAtoms; ++i) {
                const std::complex<double> R_i
                        = impurity_reflection_coefficient_discrete(
                            i, data[j].delta, g1d,
                            NAtoms, data[j].Deltac,
                            data[j].OmegaScattering,
                            kd_ensemble,
                            kL1, kL2,
                            &ensemble_scattering,
                            cphaseGateFlags);
                const RandTCoefficients rAndT
                        = ensemble_scattering(data[j].delta,
                                                kd_ensemble,
                                                kd_ensemble,
                                                data[j].Deltac,
                                                data[j].g1d,
                                                data[j].OmegaScattering,
                                                NAtoms, i);
                fileFull
                << solData[j][n].zValsWithoutScattering(i) << ";"
                << solData[j][n].zValsWithScattering(i) << ";"
                << solData[j][n].psi_after_storage0(i + NAtoms).real() << ";"
                << solData[j][n].psi_after_storage0(i + NAtoms).imag() << ";"
                << solData[j][n].psi_after_storage1(i + NAtoms).real() << ";"
                << solData[j][n].psi_after_storage1(i + NAtoms).imag() << ";"
                << solData[j][n].psi_after_scattering(i + NAtoms).real() << ";"
                << solData[j][n].psi_after_scattering(i + NAtoms).imag() << ";"
                << rAndT.r_impurity.real() << ";"
                << rAndT.r_impurity.imag() << ";"
                << rAndT.t_impurity.real() << ";"
                << rAndT.t_impurity.imag() << ";"
                << R_i.real() << ";"
                << R_i.imag() << '\n';
            }

            const int spinwaveFlags = CPHASE_GATE_PROJECT_SPINWAVES
                                      | CPHASE_GATE_ANALYTICAL_FINAL_RESULT;

            if (!((cphaseGateFlags & spinwaveFlags) == spinwaveFlags)) {
                std::stringstream nameStreamDumpFileE;
                nameStreamDumpFileE << nameStreamDumpSeed.str() << '/'
                                    << "E_fields_";
                appendWhatToVaryDataValuesToFileName(nameStreamDumpFileE, data[j], whatToVary);
                nameStreamDumpFileE << ".txt";

                std::ofstream fileE(nameStreamDumpFileE.str());
                fileE.unsetf(std::ios::floatfield);
                fileE.precision(17);
                const int num_t_with_scattering = solData[j].tVals.size();
                if (solData[j].tWeights.size() == num_t_with_scattering) {
                    fileE << "t" << ";"
                          << "E_with_scattering_Re" << ";"
                          << "E_with_scattering_Im" << ";"
                          << "E_without_scattering0_Re" << ";"
                          << "E_without_scattering0_Im" << ";"
                          << "E_without_scattering1_Re" << ";"
                          << "E_without_scattering1_Im" << ";"
                          << "quadrature_weight" << '\n';
                    for (int i = 0; i < num_t_with_scattering; ++i) {
                        fileE << solData[j].tVals(i) << ";"
                              << solData[j][n].E_with_scattering(i).real() << ";"
                              << solData[j][n].E_with_scattering(i).imag() << ";"
                              << solData[j][n].E_without_scattering0(i).real() << ";"
                              << solData[j][n].E_without_scattering0(i).imag() << ";"
                              << solData[j][n].E_without_scattering1(i).real() << ";"
                              << solData[j][n].E_without_scattering1(i).imag() << ";"
                              << solData[j].tWeights(i) << '\n';
                    }
                } else {
                    fileE << "t" << ";"
                          << "E_with_scattering_Re" << ";"
                          << "E_with_scattering_Im" << ";"
                          << "E_without_scattering0_Re" << ";"
                          << "E_without_scattering0_Im" << ";"
                          << "E_without_scattering1_Re" << ";"
                          << "E_without_scattering1_Im" << '\n';
                    for (int i = 0; i < num_t_with_scattering; ++i) {
                        fileE << solData[j].tVals(i) << ";"
                              << solData[j][n].E_with_scattering(i).real() << ";"
                              << solData[j][n].E_with_scattering(i).imag() << ";"
                              << solData[j][n].E_without_scattering0(i).real() << ";"
                              << solData[j][n].E_without_scattering0(i).imag() << ";"
                              << solData[j][n].E_without_scattering1(i).real() << ";"
                              << solData[j][n].E_without_scattering1(i).imag() << '\n';
                    }
                }
            }
        }
    }
}

std::string format_time_to_optimize(time_t seconds)
{
    int seconds_left = seconds;
    const int seconds_in_day = 86400;
    const int seconds_in_hour = 3600;
    const int seconds_in_minute = 60;
    const int days = seconds_left/seconds_in_day;
    seconds_left -= days*seconds_in_day;
    const int hours = seconds_left/seconds_in_hour;
    seconds_left -= hours*seconds_in_hour;
    const int minutes = seconds_left/seconds_in_minute;
    seconds_left -= minutes*seconds_in_minute;
    std::stringstream ret;
    if (days != 0) {
        ret << days << "d";
    }
    if (hours != 0) {
        ret << hours << "h";
    }
    if (minutes != 0) {
        ret << minutes << "m";
    }
    ret << seconds_left << "s";
    return ret.str();
}

std::vector<unsigned long int> make_random_seed_array(int numRandomSeeds)
{
    // For certain numbers of random seeds, we have
    // generated tables. Use them for consistency across
    // runs.
    std::vector<unsigned long int> randomSeeds;
    switch (numRandomSeeds) {
    case 1:
        randomSeeds = { RANDOM_SEEDS_1 };
        break;
    case 10:
        randomSeeds = { RANDOM_SEEDS_10 };
        break;
    case 100:
        randomSeeds = { RANDOM_SEEDS_100 };
        break;
    case 1000:
        randomSeeds = { RANDOM_SEEDS_1000 };
        break;
    default:
    {
        randomSeeds.resize(numRandomSeeds);
        std::random_device randomDevice;
        for (int i = 0; i < numRandomSeeds; ++i) {
            randomSeeds[i] = randomDevice();
            std::cout << randomSeeds[i] << ", ";
        }
    }
    }
    return randomSeeds;
}

std::vector<int> generate_NAtoms_array(int NAtoms_min, int NAtoms_max, int NAtoms_num)
{
    const double NAtoms_min_log = std::log10(NAtoms_min);
    const double NAtoms_max_log = std::log10(NAtoms_max);
    const double NAtoms_step_log = (NAtoms_max_log - NAtoms_min_log) / (NAtoms_num - 1);

    // When we are plotting the fidelities as functions
    // of the number of atoms we have to be careful, since
    // the atoms cannot be non-integers. On a logarithmic scale
    // it can happen that two consecutive values of the number
    // of atoms truncate to the same integer. In this case, we
    // instead take all the following points to be incremented
    // by one.
    std::vector<int> NAtoms_array;
    NAtoms_array.reserve(NAtoms_num);
    for (int i = 0; i < NAtoms_num; ++i) {
        const int NAtoms_i = static_cast<int>(std::pow(10, NAtoms_min_log + i*NAtoms_step_log));
        if (i > 0) {
            const int last_NAtoms = NAtoms_array.back();
            if (last_NAtoms < NAtoms_i) {
                NAtoms_array.push_back(NAtoms_i);
            } else if (last_NAtoms+1 <= NAtoms_max) {
                NAtoms_array.push_back(last_NAtoms+1);
            } else {
                break;
            }
        } else {
            NAtoms_array.push_back(NAtoms_i);
        }
    }
    return NAtoms_array;
}

std::vector<int> merge_NAtoms_arrays(const std::vector<int> &NAtoms_array1,
                                     const std::vector<int> &NAtoms_array2)
{
    std::vector<int> ret = NAtoms_array1;
    if (NAtoms_array2.empty()) {
        return ret;
    }
    ret.reserve(ret.size()+NAtoms_array2.size());
    if (ret[ret.size()-1] == NAtoms_array2[0]) {
        // Do not take the overlapping values
        // Note that we assume that only one value can
        // overlap (the last element of NAtoms_array1 can
        // be equal to the first element of NAtoms_array2).
        // So one just has to be a little bit careful when
        // using this function.
        ret.insert(ret.end(), NAtoms_array2.begin()+1, NAtoms_array2.end());
    } else {
        ret.insert(ret.end(), NAtoms_array2.begin(), NAtoms_array2.end());
    }
    return ret;
}

void generate_cphase_gate_fidelity_data_varied_parameter(
        double g1dInput, int NAtomsInput, double kd_ensembleInput,
        double kL1Input, double kL2Input,
        double OmegaScatteringInput, double OmegaStorageRetrievalInput,
        double scatteringTime,
        int numRandomSeeds,
        WhatToVary whatToVary, int cphaseGateFlags, int etFlagsInput,
        std::vector<int> NAtoms_array1, std::vector<int> NAtoms_array2,
        double varied_parameter_min, double varied_parameter_max,
        double varied_parameter_num)
{
    bool dumpSpinwavesAndFields = true;

    const int numExtraTNoInteractionPoints = 8;

    int etFlags = etFlagsInput;
    if (cphaseGateFlags & CPHASE_GATE_RANDOM_ATOM_POSITIONS) {
        etFlags |= ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS;
    }

    double g1d = g1dInput;
    int NAtoms = NAtomsInput;
    double OmegaScattering = OmegaScatteringInput;
    double OmegaStorageRetrieval = OmegaStorageRetrievalInput;
    double kd_ensemble = kd_ensembleInput;
    double kL1 = kL1Input;
    double kL2 = kL2Input;

    std::vector<int> NAtoms_array = merge_NAtoms_arrays(NAtoms_array1, NAtoms_array2);

    const double varied_parameter_step = (varied_parameter_max - varied_parameter_min) / (varied_parameter_num - 1);

    const double varied_parameter_min_log = std::log10(varied_parameter_min);
    const double varied_parameter_max_log = std::log10(varied_parameter_max);
    const double varied_parameter_step_log = (varied_parameter_max_log - varied_parameter_min_log) / (varied_parameter_num - 1);

    std::vector<unsigned long int> randomSeeds = make_random_seed_array(numRandomSeeds);

    if ((cphaseGateFlags & CPHASE_GATE_RANDOM_ATOM_POSITIONS) && numRandomSeeds > 10) {
        std::cout << "Turning off dumping of the extra spinwave and field data." << std::endl;
        dumpSpinwavesAndFields = false;
    }
    if (whatToVary != WhatToVary::varyKD) {
        std::cout << "kd = " << kd_ensemble;
    } else {
        std::cout << "g1d = " << g1d << ", N = " << NAtoms;
    }
    if (cphaseGateFlags & CPHASE_GATE_RANDOM_ATOM_POSITIONS) {
        std::cout << ", random atom positions (" << numRandomSeeds;
        if (numRandomSeeds == 1) {
            std::cout << " realization)";
        } else {
            std::cout << " realizations)";
        }
        if (cphaseGateFlags & CPHASE_GATE_RANDOM_POSITIONS_OPTIMIZE_WITH_REGULAR) {
            std::cout << ", but using regular atom positions for optimization";
        }
    } else {
        std::cout << ", regular atom positions";
    }
    if (etFlags & ENSEMBLE_SCATTERING_DUAL_V_ATOMS) {
        std::cout << ", Dual-V";
        if (cphaseGateFlags & CPHASE_GATE_SYMMETRIC_STORAGE) {
            std::cout << " (symmetric storage)";
        }
    } else {
        std::cout << ", Lambda-type";
    }
    if (cphaseGateFlags & CPHASE_GATE_SAGNAC_SCATTERING) {
        std::cout << ", Sagnac scattering";
    } else {
        std::cout << ", Mirror-behind-ensemble scattering";
    }
    if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_FINAL_RESULT) {
        if (cphaseGateFlags & CPHASE_GATE_PROJECT_SPINWAVES) {
            std::cout << ", projecting spinwaves (analytic storage)";
        } else {
            std::cout << ", projecting electric fields (analytic storage & retrieval)";
        }
    } else {
        if (cphaseGateFlags & CPHASE_GATE_PROJECT_SPINWAVES) {
            std::cout << ", projecting spinwaves (numeric storage)";
        } else {
            std::cout << ", projecting electric fields (numeric storage & retrieval)";
        }
    }

    if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_OPTIMIZATION) {
        std::cout << ", analytical optimization";
    } else {
        std::cout << ", numerical optimization";
    }

    if (((cphaseGateFlags & CPHASE_GATE_ANALYTICAL_OPTIMIZATION)
            || (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_FINAL_RESULT))
            && (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_STORAGE_WITH_DISPERSION_RELATION))
    {
        std::cout << " (using dispersion relation for storage)";
    }
    std::cout << std::endl;

    int num_points;
    if (whatToVary == WhatToVary::varyNAtoms) {
        num_points = NAtoms_array.size();
    } else {
        num_points = varied_parameter_num;
    }

    EnsembleScattering ensemble_scattering(etFlags);

    std::vector<CphaseGateFidelityData> data(num_points);
    std::vector<std::vector<CphaseDiagnosticData>> dataPerSeed(num_points);
    std::vector<SpinwaveAndFieldVector> solData;
    if (dumpSpinwavesAndFields) {
        solData.resize(num_points);
    }
    std::vector<ExtraTNoInteractionFidPoints> extraFidData;
    if (numExtraTNoInteractionPoints > 0) {
        extraFidData.resize(num_points);
    }
    //WARNING: This loop is not thread safe as long as we use the
    //         parameters (Delta_c and sigma) for the previous point
    //         in the optimization of the next.
    double last_Deltac = 0;
    double last_sigma = 0;
    for (int i = 0; i < num_points; ++i) {
        switch (whatToVary) {
        case WhatToVary::varyNAtoms:
            NAtoms = NAtoms_array[i];
            break;
        case WhatToVary::varyG1d:
        {
            const double g1d_over_gprime = std::pow(10, varied_parameter_min_log + i*varied_parameter_step_log);
            g1d = g1d_over_gprime / (1 + g1d_over_gprime);
            break;
        }
        case WhatToVary::varyOmegaScattering:
            OmegaScattering = std::pow(10, varied_parameter_min_log + i*varied_parameter_step_log);
            break;
        case WhatToVary::varyOmegaStorageRetrieval:
            OmegaStorageRetrieval = std::pow(10, varied_parameter_min_log + i*varied_parameter_step_log);
            break;
        case WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually:
            OmegaScattering = std::pow(10, varied_parameter_min_log + i*varied_parameter_step_log);
            OmegaStorageRetrieval = OmegaScattering;
            break;
        case WhatToVary::varyKD:
            kd_ensemble = varied_parameter_min + i*varied_parameter_step;
            break;
        case WhatToVary::varyKL1:
            kL1 = varied_parameter_min + i*varied_parameter_step;
            break;
        case WhatToVary::varyKL2:
            kL2 = varied_parameter_min + i*varied_parameter_step;
            break;
        default:
            assert(0 && "Unknown quantity to vary!");
        }
        if (cphaseGateFlags & CPHASE_GATE_VARY_OMEGA_SUCH_THAT_SCATTERING_TIME_IS_CONST) {
            OmegaScattering=std::sqrt((std::pow(g1d,3.0/2)*std::pow(NAtoms,7.0/4))/(4*std::pow(M_PI,3.0/2)*std::sqrt(1-g1d)*scatteringTime));
            OmegaStorageRetrieval=OmegaScattering;
        }
        CphaseGateFidelityParameters cphaseParameters;
        cphaseParameters.g1d = g1d;
        cphaseParameters.NAtoms = NAtoms;
        cphaseParameters.OmegaStorageRetrieval = OmegaStorageRetrieval;
        cphaseParameters.OmegaScattering = OmegaScattering;
        cphaseParameters.kd_ensemble = kd_ensemble;
        cphaseParameters.kL1 = kL1;
        cphaseParameters.kL2 = kL2;
        cphaseParameters.randomSeeds = randomSeeds;

        // Here we assume that the previous data point
        // was close enough that its Deltac and sigma
        // would be good guesses for the numerical
        // optimization of the current data point.
        // Otherwise set both of these to zero to
        // make the optimization routine try to come up
        // with reasonable guess values itself.
        cphaseParameters.Deltac = last_Deltac;
        cphaseParameters.sigma = last_sigma;

        CphaseFidelities fid;
        CphaseFidelities fid_tNoInteraction_one;
        SpinwaveAndFieldVector sol(numRandomSeeds);
        std::vector<CphaseDiagnosticData> diagData(numRandomSeeds);
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        time_t seconds_start = ts.tv_sec;
        calculate_cphase_fidelities_numerical_for_optimal_Deltac_sigma(
                    &fid, &fid_tNoInteraction_one, sol, diagData,
                    &cphaseParameters, cphaseGateFlags,
                    &ensemble_scattering);
        clock_gettime(CLOCK_MONOTONIC, &ts);
        time_t seconds_stop = ts.tv_sec;

        last_Deltac = cphaseParameters.Deltac;
        last_sigma = cphaseParameters.sigma;

        data[i].P_success = fid.P_success;
        data[i].F_swap = fid.F_swap;
        data[i].F_CJ = fid.F_CJ;
        data[i].F_CJ_conditional = fid.F_CJ_conditional;

        data[i].P_success_tNoInteraction_one
                = fid_tNoInteraction_one.P_success;
        data[i].F_swap_tNoInteraction_one
                = fid_tNoInteraction_one.F_swap;
        data[i].F_CJ_tNoInteraction_one
                = fid_tNoInteraction_one.F_CJ;
        data[i].F_CJ_conditional_tNoInteraction_one
                = fid_tNoInteraction_one.F_CJ_conditional;
        data[i].single_photon_storage_retrieval_eff
                = fid.single_photon_storage_retrieval_eff;

        data[i].kd_ensemble = kd_ensemble;
        data[i].kL1 = kL1;
        data[i].kL2 = kL2;
        data[i].g1d = g1d;
        data[i].NAtoms = NAtoms;
        data[i].delta = cphaseParameters.delta;
        data[i].Deltac = cphaseParameters.Deltac;
        data[i].OmegaScattering = OmegaScattering;
        data[i].OmegaStorageRetrieval = OmegaStorageRetrieval;
        data[i].sigma = cphaseParameters.sigma;
        data[i].tNoInteractionAbs = cphaseParameters.tNoInteractionAbs;
        data[i].tNoInteractionArg = cphaseParameters.tNoInteractionArg;
        data[i].t_storage = cphaseParameters.t_storage;
        data[i].t_retrieval = cphaseParameters.t_retrieval;
        data[i].t_to_pass_ensemble = cphaseParameters.t_to_pass_ensemble;

        dataPerSeed[i] = diagData;
        if (dumpSpinwavesAndFields) {
            solData[i] = sol;
        }
        if (numExtraTNoInteractionPoints > 0) {
            const int totExtraPoints = numExtraTNoInteractionPoints+2;
            ExtraTNoInteractionFidPoints extraFidPoints;
            extraFidPoints.tNoInteraction.resize(totExtraPoints);
            extraFidPoints.fid.resize(totExtraPoints);
            const std::complex<double> I(0,1);
            const std::complex<double> optimalTNoInteraction
                    = cphaseParameters.tNoInteractionAbs
                      *std::exp(I*cphaseParameters.tNoInteractionArg);
            const int sol_size = sol.size();
            std::vector<std::complex<double>> noImpR(sol_size);
            for (int j = 0; j < sol_size; ++j) {
                noImpR[j] = diagData[j].noImpurityR;
            }
            std::complex<double> stepTNoInteraction
                    = (1.0-optimalTNoInteraction)
                      /static_cast<double>(totExtraPoints - 1);
            bool onlyAnalyticalCalculation;
            if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_FINAL_RESULT) {
                onlyAnalyticalCalculation = true;
            } else {
                onlyAnalyticalCalculation = false;
            }
            for (int j = 0; j < totExtraPoints; ++j) {
                CphaseFidelities cur_fid;
                const std::complex<double> cur_tNoInteraction
                        = optimalTNoInteraction
                          +static_cast<double>(j)*stepTNoInteraction;
                calculate_cphase_fidelities_for_tNoInteraction(
                        sol, &cur_fid, noImpR, cur_tNoInteraction,
                        cphaseGateFlags, onlyAnalyticalCalculation);
                extraFidPoints.fid[j] = cur_fid;
                extraFidPoints.tNoInteraction[j] = cur_tNoInteraction;
            }
            extraFidData[i] = extraFidPoints;
        }
        if (whatToVary != WhatToVary::varyKD) {
            std::cout << "  g1d = " << g1d << ", NAtoms = " << NAtoms;
            if (cphaseGateFlags & CPHASE_GATE_VARY_OMEGA_SUCH_THAT_SCATTERING_TIME_IS_CONST) {
                std::cout << ", Omega (Scattering & StorageRetrieval) = " << OmegaScattering;
            }
        } else {
            std::cout << "  kd = " << kd_ensemble;
        }
        if (whatToVary == WhatToVary::varyOmegaScattering) {
            std::cout << ", OmegaScattering = " << OmegaScattering;
        }
        if (whatToVary == WhatToVary::varyOmegaStorageRetrieval) {
            std::cout << ", OmegaStorageRetrieval = " << OmegaStorageRetrieval;
        }
        if (whatToVary == WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually) {
            std::cout << ", Omega (Scattering & StorageRetrieval) = " << OmegaScattering;
        }
        if (whatToVary == WhatToVary::varyKL1) {
            std::cout << ", kL1 = " << kL1
                      << ", arg(t_b) = " << data[i].tNoInteractionArg;
        }
        if (whatToVary == WhatToVary::varyKL2) {
            std::cout << ", kL2 = " << kL2;
        }
        std::cout << ", F_CJ (t_{no int}=1) = " << fid_tNoInteraction_one.F_CJ
                  << ", F_CJ_cond (t_{no int}=1) = " << fid_tNoInteraction_one.F_CJ_conditional
                  << ", F_CJ = " << fid.F_CJ
                  << ", F_CJ_cond = " << fid.F_CJ_conditional
                  << ", time to optimize = "
                  << format_time_to_optimize(seconds_stop - seconds_start)
                  << " (" << seconds_stop - seconds_start << "s)"
                  << std::endl;
    }

    std::stringstream nameStreamBase;
    nameStreamBase << "cphase_fidelity_";
    nameStreamBase << whatToVaryToFileName(whatToVary);

    if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_FINAL_RESULT) {
        if (cphaseGateFlags & CPHASE_GATE_PROJECT_SPINWAVES) {
            nameStreamBase << "_aS";
        } else {
            nameStreamBase << "_aE";
        }
    } else {
        if (cphaseGateFlags & CPHASE_GATE_PROJECT_SPINWAVES) {
            nameStreamBase << "_nS";
        } else {
            nameStreamBase << "_nE";
        }
    }
    if (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_OPTIMIZATION) {
        nameStreamBase << "_a_opt";
    } else {
        nameStreamBase << "_n_opt";
    }
    if (((cphaseGateFlags & CPHASE_GATE_ANALYTICAL_OPTIMIZATION)
            || (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_FINAL_RESULT))
            && (cphaseGateFlags & CPHASE_GATE_ANALYTICAL_STORAGE_WITH_DISPERSION_RELATION))
    {
        nameStreamBase << "_disp_storage";
    }
    if (etFlags & ENSEMBLE_SCATTERING_DUAL_V_ATOMS) {
        nameStreamBase << "_dualv";
        if (cphaseGateFlags & CPHASE_GATE_SYMMETRIC_STORAGE) {
            nameStreamBase << "_sym";
        }
    } else {
        nameStreamBase << "_lambda";
    }
    if (cphaseGateFlags & CPHASE_GATE_SAGNAC_SCATTERING) {
        nameStreamBase << "_sagnac";
    }
    switch (whatToVary) {
    case WhatToVary::varyNAtoms:
        nameStreamBase << "_g1d_" << g1d;
        if (cphaseGateFlags & CPHASE_GATE_VARY_OMEGA_SUCH_THAT_SCATTERING_TIME_IS_CONST) {
            nameStreamBase << "_scatteringTime_" << scatteringTime;
        } else {
            nameStreamBase << "_OmegaScattering_" << OmegaScattering
                           << "_OmegaStorageRetrieval_" << OmegaStorageRetrieval;
        }
        nameStreamBase << "_kd_" << kd_ensemble;
        if (kL1Input != 0) {
            nameStreamBase << "_kL1_" << kL1Input;
        }
        if (kL2Input != 0) {
            nameStreamBase << "_kL2_" << kL2Input;
        }
        break;
    case WhatToVary::varyG1d:
        nameStreamBase << "_N_" << NAtoms
                       << "_OmegaScattering_" << OmegaScattering
                       << "_OmegaStorageRetrieval_" << OmegaStorageRetrieval
                       << "_kd_" << kd_ensemble;
        if (kL1Input != 0) {
            nameStreamBase << "_kL1_" << kL1Input;
        }
        if (kL2Input != 0) {
            nameStreamBase << "_kL2_" << kL2Input;
        }
        break;
    case WhatToVary::varyOmegaScattering:
        nameStreamBase << "_g1d_" << g1d
                       << "_N_" << NAtoms
                       << "_OmegaStorageRetrieval_" << OmegaStorageRetrieval
                       << "_kd_" << kd_ensemble;
        if (kL1Input != 0) {
            nameStreamBase << "_kL1_" << kL1Input;
        }
        if (kL2Input != 0) {
            nameStreamBase << "_kL2_" << kL2Input;
        }
        break;
    case WhatToVary::varyOmegaStorageRetrieval:
        nameStreamBase << "_g1d_" << g1d
                       << "_N_" << NAtoms
                       << "_OmegaScattering_" << OmegaScattering
                       << "_kd_" << kd_ensemble;
        if (kL1Input != 0) {
            nameStreamBase << "_kL1_" << kL1Input;
        }
        if (kL2Input != 0) {
            nameStreamBase << "_kL2_" << kL2Input;
        }
        break;
    case WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually:
        nameStreamBase << "_g1d_" << g1d
                       << "_N_" << NAtoms
                       << "_kd_" << kd_ensemble;
        if (kL1Input != 0) {
            nameStreamBase << "_kL1_" << kL1Input;
        }
        if (kL2Input != 0) {
            nameStreamBase << "_kL2_" << kL2Input;
        }
        break;
    case WhatToVary::varyKD:
        nameStreamBase << "_g1d_" << g1d
                       << "_N_" << NAtoms
                       << "_OmegaScattering_" << OmegaScattering
                       << "_OmegaStorageRetrieval_" << OmegaStorageRetrieval;
        if (kL1Input != 0) {
            nameStreamBase << "_kL1_" << kL1Input;
        }
        if (kL2Input != 0) {
            nameStreamBase << "_kL2_" << kL2Input;
        }
        break;
    case WhatToVary::varyKL1:
        nameStreamBase << "_g1d_" << g1d
                       << "_N_" << NAtoms
                       << "_OmegaScattering_" << OmegaScattering
                       << "_OmegaStorageRetrieval_" << OmegaStorageRetrieval
                       << "_kd_" << kd_ensemble
                       << "_kL2_" << kL2Input;
        break;
    case WhatToVary::varyKL2:
        nameStreamBase << "_g1d_" << g1d
                       << "_N_" << NAtoms
                       << "_OmegaScattering_" << OmegaScattering
                       << "_OmegaStorageRetrieval_" << OmegaStorageRetrieval
                       << "_kd_" << kd_ensemble
                       << "_kL1_" << kL1Input;
        break;
    default:
        assert(0 && "Unknown quantity to vary!");
    }
    if (cphaseGateFlags & CPHASE_GATE_RANDOM_ATOM_POSITIONS) {
        nameStreamBase << "_numSeeds_" << numRandomSeeds;
        if (cphaseGateFlags & CPHASE_GATE_RANDOM_POSITIONS_OPTIMIZE_WITH_REGULAR) {
            nameStreamBase << "_reg_opt";
        }
    }
    std::stringstream nameStreamDump;
    const std::string nameStreamBaseStr = nameStreamBase.str();
    nameStreamDump << nameStreamBaseStr << "_data";
    const std::string nameStreamDumpStr = nameStreamDump.str();
    std::stringstream nameStream;
    nameStream << nameStreamBaseStr << ".txt";
    const bool appendOmegaValues
            = cphaseGateFlags & CPHASE_GATE_VARY_OMEGA_SUCH_THAT_SCATTERING_TIME_IS_CONST;
    write_cphase_gate_fidelity_data_to_file(nameStream.str(), data, whatToVary,
                                            appendOmegaValues);

    if (numExtraTNoInteractionPoints > 0) {
        std::stringstream nameStreamExtra;
        nameStreamExtra << nameStreamBaseStr << "_extra_"
                        << numExtraTNoInteractionPoints << ".txt";

        write_cphase_gate_fidelity_extra_tNoInteraction_data_to_file(
                    nameStreamExtra.str(), data, extraFidData, whatToVary,
                    appendOmegaValues);
    }

    int err = rmrf(nameStreamDumpStr.c_str());

    err = mkdir(nameStreamDumpStr.c_str(), 0777);
    bool dumpDataDirExists = true;
    if (err == -1 && errno != EEXIST) {
        std::cout << "Failed making dump data directory: "
                  << strerror(errno) << std::endl;
        dumpDataDirExists = false;
    }
    if (dumpDataDirExists) {
        write_cphase_gate_fidelity_seed_specific_data_to_file(
                    nameStreamDumpStr, randomSeeds, data, dataPerSeed, solData,
                    ensemble_scattering, cphaseGateFlags,
                    dumpSpinwavesAndFields, whatToVary);
    }
}

void generate_eit_models_comparison_plot_data(double g1d,
                                              int NAtoms,
                                              double kd_ensemble,
                                              double OmegaStorageRetrieval,
                                              double sigma,
                                              unsigned long int randomSeed,
                                              int etFlags)
{
    HamiltonianParams eitParams;
    eitParams.g1d = g1d;
    eitParams.NAtoms = NAtoms;
    eitParams.Omega = OmegaStorageRetrieval;
    eitParams.Deltac = 0;
    eitParams.delta = 0;
    eitParams.counterpropagating = false;
    eitParams.putPhasesInTheClassicalDrive = false;
    eitParams.evolutionMethod = EvolutionMethod::RK4;
    eitParams.kd_ensemble = kd_ensemble;

    const double L = 1;
    eitParams.gridSpacing = grid_spacing(eitParams.NAtoms, L);

    const double v_g_eit = 2*eitParams.gridSpacing*pow(eitParams.Omega, 2)/eitParams.g1d;
    const double alphaI = 4.0*(1-eitParams.g1d)*pow(eitParams.gridSpacing*eitParams.Omega/eitParams.g1d, 2);
    const double timeToPassEnsemble = 1.0/v_g_eit;
    const double initialEITPropagation = 0.5;
    const double initialMean = 0.0;
    const double t_EIT = initialEITPropagation*timeToPassEnsemble;
    const double sigmaInitial = sigma/std::sqrt(1+alphaI*t_EIT/(2*sigma*sigma));

    eitParams.input_electric_field_R_factor = 1;
    eitParams.inputE_width = sigmaInitial/v_g_eit;
    eitParams.inputE_mean = INPUT_E_MEAN_IN_UNITS_OF_WIDTH*eitParams.inputE_width;
    if (etFlags & ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS) {
        eitParams.randomAtomPositions = true;
    }

    const double t_free_space = std::abs(eitParams.inputE_mean);
    const double t_storage = t_EIT + t_free_space;

    const double storageTolAbs = STORED_SPIN_WAVE_TOLERANCE_ABS;
    const double storageTolRel = STORED_SPIN_WAVE_TOLERANCE_REL;

    // We only need this to find the positions of the atoms
    // (This is the same way we do it for the actual
    // fidelity calculations).
    EnsembleScattering ensemble_scattering(etFlags);
    ensemble_scattering.setRandomSeed(randomSeed);
    ensemble_scattering.fillAtomArraysNewValues(eitParams.NAtoms,
                                                  kd_ensemble, kd_ensemble);
    eitParams.atom_positions = ensemble_scattering.atomPositions();

    const std::vector<std::complex<double>> S_dispersion_relation
            = calculate_stored_spinwave_eit_dispersion_relation_plus(eitParams,
                                                                     true);
    const std::vector<std::complex<double>> S_kernel
            = calculate_stored_spinwave_eit_kernel_plus(eitParams, t_storage,
                                                        storageTolAbs,
                                                        storageTolRel,
                                                        true);

    const bool onlyAnalyticalCalculation = false;
    double normThreshold;
    double fidThreshold;
    double dtInitial;
    set_stored_spin_wave_set_tolerances(dtInitial, normThreshold,
                                        fidThreshold,
                                        onlyAnalyticalCalculation);
    const EvolutionMethod method
            = EvolutionMethod::Default;
    int flags = 0;
    flags |= LAMBDA_HAMILTONIAN_EVOLVE_CALCULATE_ELECTRIC_FIELD;

    LambdaHamiltonian1Excitation H_EIT;
    H_EIT.setParams(eitParams);
    Eigen::VectorXcd psi_numerical
            = calculate_stored_spinwave_eit_field_elimination(H_EIT, method, t_storage, dtInitial, normThreshold, flags);

    std::stringstream nameStream;
    nameStream << "eit_storage_models_comparison"
               << "_g1d_" << eitParams.g1d
               << "_NAtoms_" << eitParams.NAtoms
               << "_OmegaStorageRetrieval_" << eitParams.Omega
               << "_kd_" << kd_ensemble;
    if (etFlags & ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS) {
        nameStream << "_seed_" << randomSeed;
    }
    nameStream << ".txt";

    std::ofstream file(nameStream.str());
    // If an IEEE 754 double precision is converted to a
    // decimal string with at least 17 significant digits
    // and then converted back to double, then the final
    // number must match the original.
    file.precision(17);

    file << "z" << ';'
         << "S_dispersion_relation_re" << ';'
         << "S_dispersion_relation_im" << ';'
         << "S_kernel_re" << ';'
         << "S_kernel_im" << ';'
         << "S_discrete_re" << ';'
         << "S_discrete_im" << '\n';

    const double dz = 1.0/eitParams.NAtoms;
    const double dzSqrt = std::sqrt(dz);
    for (int i = 0; i < eitParams.NAtoms; ++i) {
        file << eitParams.atom_positions[i] << ';'
             << S_dispersion_relation[i].real()*dzSqrt << ';'
             << S_dispersion_relation[i].imag()*dzSqrt << ';'
             << S_kernel[i].real()*dzSqrt << ';'
             << S_kernel[i].imag()*dzSqrt << ';'
             << psi_numerical(i + eitParams.NAtoms).real() << ';'
             << psi_numerical(i + eitParams.NAtoms).imag() << '\n';
    }
    std::cout << "Wrote to: " << nameStream.str() << std::endl;
}

void generate_article_plot_data()
{
    // Running all of the function calls below will take a lot of time
    // it is best to comment all but one of them at a time. (Each of these
    // function calls outputs separate data files, so they can be run in
    // parallel, but then use the OpenMP variable "OMP_NUM_THREADS"
    // appropriately such as not to oversubscribe the available CPU cores.)
    //
    // The complete list is here mostly for the documentation purposes.

    const double g1d_default = 0.05;
    const int NAtoms_default = 10000;

    const double OmegaScattering_default = 1;
    const double OmegaStorageRetrieval_default = 1;
    const double scatteringTime_default = 50;

    const double kL1_default = 0;
    const double kL2_default = 0;

    // This is just a dummy value for use in most of
    // the cases below, where we assume regularly placed
    // atoms. The underlying code does not care for the
    // precise value as long as the flag
    // "CPHASE_GATE_RANDOM_ATOM_POSITIONS"
    // is not set.
    const int numRandomSeeds_single = 1;

    const double kd_ensemble_min = 0.001;
    const double kd_ensemble_max = 0.999;
    const int kd_ensemble_num = 999;

    const double OmegaScattering_min = 1;
    const double OmegaScattering_max = 1e2;
    const int OmegaScattering_num = 201;

    // Generates data file used in Fig. 3 of the main article and 
    // Figs. S3 and S10 of the Supplemental Material.
    //
    // Lambda-type scheme data; fixed $\Gamma_{1D}/\Gamma=0.05$, $kd/\pi=0.5$,
    // and varied NAtoms.
    //
    // Here we go up to the maximum NAtoms=10^6 to have a better idea of the
    // asymptotic behavior.
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyNAtoms,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                0,
                generate_NAtoms_array(1e3, 1e4, 1001),
                generate_NAtoms_array(1e4, 1e6, 21),
                0, 0, 0);

    // Generates data file used in Fig. S7 of the Supplemental Material.
    //
    // Below, we do the final computation using
    // storage and retrieval with the fully discrete
    // storage, but the optimization is still using the
    // discretized storage and retrieval kernels.
    //
    // The maximum NAtoms is reduced to 10^4, since otherwise, it will take too
    // long to compute
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyNAtoms,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                0,
                generate_NAtoms_array(1e3, 1e4, 11),
                {},
                0, 0, 0);

    // Generates data file used in Fig. 3 of the main article.
    //
    // Dual-V scheme data; fixed $\Gamma_{1D}/\Gamma=0.05$, $kd/\pi=0.266$,
    // and varied NAtoms
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.266, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyNAtoms,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                ENSEMBLE_SCATTERING_DUAL_V_ATOMS,
                generate_NAtoms_array(1e3, 1e4, 1001),
                generate_NAtoms_array(1e4, 1e5, 11),
                0, 0, 0);

    // Generates data file used in Fig. S11 of the Supplemental Material.
    //
    // Dual-V scheme data; fixed $\Gamma_{1D}/\Gamma=0.05$, NAtoms=10000,
    // and varied $kd/\pi$
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.266, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyKD,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                ENSEMBLE_SCATTERING_DUAL_V_ATOMS,
                {},
                {},
                kd_ensemble_min, kd_ensemble_max, kd_ensemble_num);

    // Generates data file used in Fig. S12 of the Supplemental Material.
    //
    // Dual-V scheme data; fixed $\Gamma_{1D}/\Gamma=0.05$, $kd/\pi=0.266$,
    // varied NAtoms and random atom positions (100 realizations)
    const int numRandomSeeds_many = 100;
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.266, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_many,
                WhatToVary::varyNAtoms,
                CPHASE_GATE_RANDOM_ATOM_POSITIONS |
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                ENSEMBLE_SCATTERING_DUAL_V_ATOMS,
                generate_NAtoms_array(1e3, 1e4, 101),
                {},
                0, 0, 0);

    // Generates data file used in Fig. S9 of the Supplemental Material.
    //
    // Lambda-type scheme data; fixed NAtoms = 10000, $kd/\pi=0.5$,
    // $\Gamma_{1D}/\Gamma=0.05$ and varied OmegaScattering and
    // OmegaStorageRetrieval.
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                0,
                {},
                {},
                OmegaScattering_min, OmegaScattering_max, OmegaScattering_num);

    // Generates data file used in Fig. S9 of the Supplemental Material.
    //
    // Dual-V scheme data; fixed NAtoms = 10000, $kd/\pi=0.266$,
    // $\Gamma_{1D}/\Gamma=0.05$ and varied OmegaScattering and
    // OmegaStorageRetrieval.
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.266, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyOmegaScatteringAndStorageRetrievalEqually,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                ENSEMBLE_SCATTERING_DUAL_V_ATOMS,
                {},
                {},
                OmegaScattering_min, OmegaScattering_max, OmegaScattering_num);

    // Generates data file used in Fig. S8 of the Supplemental Material.
    //
    // Lambda-type scheme data; fixed NAtoms = 10000, $kd/\pi=0.5$,
    // and varied kL1
    const double kL1_min = -0.5;
    const double kL1_max = 0.5;
    const int kL1_num = 101;
    generate_cphase_gate_fidelity_data_varied_parameter(
                0.05, 10000, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyKL1,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                0,
                {},
                {},
                kL1_min, kL1_max, kL1_num);

    // Generates data file used in Fig. S8 of the Supplemental Material.
    //
    // Dual-V scheme data; fixed NAtoms = 10000, $kd/\pi=0.5$,
    // and varied kL1
    generate_cphase_gate_fidelity_data_varied_parameter(
                0.05, 10000, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyKL1,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                ENSEMBLE_SCATTERING_DUAL_V_ATOMS,
                {},
                {},
                kL1_min, kL1_max, kL1_num);

    // Generates data files used in Fig. S6 of the Supplemental Material.
    //
    // Regular placement
    generate_eit_models_comparison_plot_data(
                g1d_default, NAtoms_default, 0.266,
                OmegaStorageRetrieval_default, 0.1, 12345,
                0);
    // Random placement
    generate_eit_models_comparison_plot_data(
                g1d_default, NAtoms_default, 0.266,
                OmegaStorageRetrieval_default, 0.1, 12345,
                ENSEMBLE_SCATTERING_RANDOM_ATOM_POSITIONS);
}

void generate_extra_plot_data()
{
    // This a collection of plots, which are not part of the
    // article.

    const double g1d_default = 0.05;
    const int NAtoms_default = 10000;

    const double OmegaScattering_default = 1;
    const double OmegaStorageRetrieval_default = 1;
    const double scatteringTime_default = 50;

    const double kL1_default = 0;
    const double kL2_default = 0;

    const double OmegaScattering_min = 1;
    const double OmegaScattering_max = 1e2;
    const int OmegaScattering_num = 201;

    // This is just a dummy value for use in most of
    // the cases below, where we assume regularly placed
    // atoms. The underlying code does not care for the
    // precise value as long as the flag
    // "CPHASE_GATE_RANDOM_ATOM_POSITIONS"
    // is not set.
    const int numRandomSeeds_single = 1;

    // Lambda-type scheme data; fixed $\Gamma_{1D}/\Gamma=0.05$, $kd/\pi=0.5$,
    // and varied NAtoms.
    // OmegaScattering and OmegaStorageRetrieval are varied
    // such that the scattering time (only dependent on OmegaScattering, but we
    // set OmegaStorageRetrieval to the same value for simplicity)
    //
    // Here we go up to the maximum NAtoms=10^6 to have a better idea of the
    // asymptotic behavior, when we compare the analytical expressions to the
    // numerics in the Supplemental Material of the article.
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyNAtoms,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING |
                CPHASE_GATE_VARY_OMEGA_SUCH_THAT_SCATTERING_TIME_IS_CONST,
                0,
                generate_NAtoms_array(1e3, 1e4, 1001),
                generate_NAtoms_array(1e4, 1e6, 21),
                0, 0, 0);

    // Below, we do the final computation using
    // storage and retrieval with the fully discrete
    // storage, but the optimization is still using the
    // discretized storage and retrieval kernels.
    //
    // The maximum NAtoms is reduced to 10^4, since otherwise, it will take too
    // long to compute
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyNAtoms,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING |
                CPHASE_GATE_VARY_OMEGA_SUCH_THAT_SCATTERING_TIME_IS_CONST,
                0,
                generate_NAtoms_array(1e3, 1e4, 11),
                {},
                0, 0, 0);

    // Dual-V scheme data; fixed $\Gamma_{1D}/\Gamma=0.05$, $kd/\pi=0.266$,
    // and varied NAtoms
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.266, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyNAtoms,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING |
                CPHASE_GATE_VARY_OMEGA_SUCH_THAT_SCATTERING_TIME_IS_CONST,
                ENSEMBLE_SCATTERING_DUAL_V_ATOMS,
                generate_NAtoms_array(1e3, 1e4, 1001),
                generate_NAtoms_array(1e4, 1e5, 11),
                0, 0, 0);


    // Lambda-type scheme data; fixed NAtoms = 1000, $kd/\pi=0.5$,
    // and varied $\Gamma_{1D}/\Gamma$. Note: the minimium and
    // maximum values (0.1 and 100 respectively) are interpreted
    // as $\Gamma_{1D}/\Gamma'$ (notice it's Gamma' instead of Gamma!).
    const double g1d_over_gprime_min = 0.1;
    const double g1d_over_gprime_max = 100;
    const int g1d_num = 31;
    generate_cphase_gate_fidelity_data_varied_parameter(
                0.1, 1000, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyG1d,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                0,
                {},
                {},
                g1d_over_gprime_min, g1d_over_gprime_max, g1d_num);

    // Lambda-type scheme data; fixed NAtoms = 10000, $kd/\pi=0.5$,
    // $\Gamma_{1D}/\Gamma=0.05$ and varied OmegaScattering.
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.5, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyOmegaStorageRetrieval,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                0,
                {},
                {},
                OmegaScattering_min, OmegaScattering_max, OmegaScattering_num);

    // Dual-V scheme data; fixed NAtoms = 10000, $kd/\pi=0.266$,
    // $\Gamma_{1D}/\Gamma=0.05$ and varied OmegaScattering.
    generate_cphase_gate_fidelity_data_varied_parameter(
                g1d_default, NAtoms_default, 0.266, kL1_default, kL2_default,
                OmegaScattering_default, OmegaStorageRetrieval_default,
                scatteringTime_default,
                numRandomSeeds_single,
                WhatToVary::varyOmegaStorageRetrieval,
                CPHASE_GATE_ANALYTICAL_OPTIMIZATION |
                CPHASE_GATE_ANALYTICAL_FINAL_RESULT |
                CPHASE_GATE_SYMMETRIC_STORAGE |
                CPHASE_GATE_SAGNAC_SCATTERING,
                ENSEMBLE_SCATTERING_DUAL_V_ATOMS,
                {},
                {},
                OmegaScattering_min, OmegaScattering_max, OmegaScattering_num);
}
} // unnamed namespace

int main(int argc, char *argv[])
{
    generate_article_plot_data();
    //generate_extra_plot_data();
    return 0;
}
