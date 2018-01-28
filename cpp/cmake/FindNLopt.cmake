find_package(PkgConfig)

if (PKG_CONFIG_FOUND)
    pkg_check_modules(PC_NLOPT QUIET nlopt)
endif(PKG_CONFIG_FOUND)

#There is also the "libnlopt_cxx.so" library for C++,
#but the only difference is the addition of the StoGO algorithm.
find_path(NLOPT_INCLUDE_DIR nlopt.h HINTS ${PC_NLOPT_INCLUDE_DIRS})

find_library(NLOPT_LIBRARY NAMES nlopt HINTS ${PC_NLOPT_LIBRARY_DIRS})

set(NLOPT_LIBRARIES ${NLOPT_LIBRARY} )
set(NLOPT_INCLUDE_DIRS ${NLOPT_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    NLopt DEFAULT_MSG
    NLOPT_LIBRARIES NLOPT_INCLUDE_DIR
)

mark_as_advanced(
    NLOPT_INCLUDE_DIR
    NLOPT_LIBRARY
)
