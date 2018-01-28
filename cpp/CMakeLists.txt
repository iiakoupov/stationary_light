cmake_minimum_required(VERSION 2.8.8)
project(stationary_light)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(NLopt REQUIRED)
include_directories(${NLOPT_INCLUDE_DIR})

#The default CMAKE_CXX_FLAGS_RELEASE is "-O3 -DNDEBUG".
#Thus we are being a little more conservative here
#and set only -DNDEBUG, while "-O2" is added to all
#build types via CMAKE_CXX_FLAGS.
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
set(CMAKE_CXX_FLAGS "-std=c++0x -march=native -O2 -fopenmp")

add_subdirectory(quadrature)

set(bandstructure_SRCS
    bandstructure.cpp
)

set(cphase_gate_SRCS
    cphase_gate.cpp
)

set(cphase_gate_plots_SRCS
    cphase_gate_plots.cpp
)

set(dispersion_relation_plots_SRCS
    dispersion_relation_plots.cpp
)

set(ensemble_scattering_SRCS
    ensemble_scattering.cpp
)

set(findroot_SRCS
    findroot.cpp
)

set(gate_common_SRCS
    gate_common.cpp
)

set(lambda_hamiltonian_SRCS
    lambda_hamiltonian.cpp
)

set(urandom_SRCS
    urandom.cpp
)

#Libraries

add_library(bandstructure ${bandstructure_SRCS})
target_link_libraries(bandstructure urandom)

add_library(cphase_gate ${cphase_gate_SRCS})
target_link_libraries(cphase_gate lambda_hamiltonian ensemble_scattering
    urandom gate_common quad ${NLOPT_LIBRARIES})

add_library(ensemble_scattering ${ensemble_scattering_SRCS})
target_link_libraries(ensemble_scattering urandom)

add_library(findroot ${findroot_SRCS})

add_library(gate_common ${gate_common_SRCS})
target_link_libraries(gate_common bandstructure)

add_library(lambda_hamiltonian ${lambda_hamiltonian_SRCS})
target_link_libraries(lambda_hamiltonian urandom)

add_library(urandom ${urandom_SRCS})
target_link_libraries(urandom)

#Executables

add_executable(cphase_gate_plots ${cphase_gate_plots_SRCS})
target_link_libraries(cphase_gate_plots cphase_gate)

add_executable(dispersion_relation_plots
    ${dispersion_relation_plots_SRCS})
target_link_libraries(dispersion_relation_plots bandstructure
    ensemble_scattering findroot)

