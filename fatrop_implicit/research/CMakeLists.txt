find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)

add_executable(global_parameter_kkt_benchmark
  global_parameter_kkt_benchmark.cpp
)

add_executable(dtoc_fatrop_benchmark
  dtoc_fatrop_benchmark.cpp
)

target_link_libraries(global_parameter_kkt_benchmark
  PRIVATE
    fatrop
    Eigen3::Eigen
    PkgConfig::IPOPT
)

target_link_libraries(dtoc_fatrop_benchmark PRIVATE fatrop)

set(CUTEST_ROOT "" CACHE PATH "Prefix of an optional CUTEst installation")
if(NOT CUTEST_ROOT AND
   EXISTS "${CMAKE_SOURCE_DIR}/../research-deps/install/include/cutest_c.h")
  set(CUTEST_ROOT "${CMAKE_SOURCE_DIR}/../research-deps/install")
endif()

find_path(CUTEST_INCLUDE_DIR
  NAMES cutest_c.h
  HINTS "${CUTEST_ROOT}/include"
)
find_library(CUTEST_DOUBLE_LIBRARY
  NAMES cutest_double
  HINTS "${CUTEST_ROOT}/lib"
)

if(CUTEST_INCLUDE_DIR AND CUTEST_DOUBLE_LIBRARY)
  add_executable(cutest_probe cutest_probe.cpp)
  target_include_directories(cutest_probe PRIVATE "${CUTEST_INCLUDE_DIR}")
  target_link_libraries(cutest_probe PRIVATE "${CUTEST_DOUBLE_LIBRARY}")
  add_executable(cutest_ipopt_benchmark cutest_ipopt_benchmark.cpp)
  target_include_directories(cutest_ipopt_benchmark
    PRIVATE "${CUTEST_INCLUDE_DIR}"
  )
  target_link_libraries(cutest_ipopt_benchmark
    PRIVATE
      "${CUTEST_DOUBLE_LIBRARY}"
      PkgConfig::IPOPT
  )
  set_target_properties(cutest_probe cutest_ipopt_benchmark PROPERTIES
    BUILD_RPATH "${CUTEST_ROOT}/lib"
  )
else()
  message(STATUS
    "CUTEst not found; CUTEst probe and IPOPT adapter will not be built"
  )
endif()
