find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)

add_executable(global_parameter_kkt_benchmark
  global_parameter_kkt_benchmark.cpp
)

add_executable(parametric_riccati_reference
  parametric_riccati_reference.cpp
)

add_executable(heterogeneous_phase_kkt_benchmark
  heterogeneous_phase_kkt_benchmark.cpp
)

add_executable(interval_scoped_kkt_benchmark
  interval_scoped_kkt_benchmark.cpp
)

add_executable(interval_scoped_ocp_benchmark
  interval_scoped_ocp_benchmark.cpp
)

add_executable(rank_deficient_implicit_kkt_benchmark
  rank_deficient_implicit_kkt_benchmark.cpp
)

add_executable(nonlinear_global_parameter_sqp_benchmark
  nonlinear_global_parameter_sqp_benchmark.cpp
)

add_executable(nonlinear_interval_scoped_sqp_benchmark
  nonlinear_interval_scoped_sqp_benchmark.cpp
  interval_scoped_collocation_problem.cpp
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

target_link_libraries(parametric_riccati_reference
  PRIVATE
    Eigen3::Eigen
)

target_link_libraries(heterogeneous_phase_kkt_benchmark
  PRIVATE
    fatrop
    Eigen3::Eigen
)

target_link_libraries(interval_scoped_kkt_benchmark
  PRIVATE
    fatrop
    Eigen3::Eigen
)

target_link_libraries(interval_scoped_ocp_benchmark
  PRIVATE
    fatrop
    Eigen3::Eigen
)

target_link_libraries(rank_deficient_implicit_kkt_benchmark
  PRIVATE
    fatrop
    Eigen3::Eigen
)

target_link_libraries(nonlinear_global_parameter_sqp_benchmark
  PRIVATE
    fatrop
    Eigen3::Eigen
    PkgConfig::IPOPT
)

target_link_libraries(nonlinear_interval_scoped_sqp_benchmark
  PRIVATE
    fatrop
    Eigen3::Eigen
    PkgConfig::IPOPT
)

add_test(
  NAME nonlinear_interval_scoped_collocation_regression
  COMMAND nonlinear_interval_scoped_sqp_benchmark
    --phase-nodes 4,5,4,5
    --phase-nx 2,3,2,4
    --phase-nu 1,1,2,1
    --phase-parameters 1
    --interface-parameters 1
    --segment-parameters 1
    --segment-width 2
    --segment-stride 1
    --global-parameters 1
    --collocation-degree 3
    --repeats 1
)
set_tests_properties(
  nonlinear_interval_scoped_collocation_regression
  PROPERTIES TIMEOUT 30
)

add_test(
  NAME nonlinear_interval_scoped_ipm_regression
  COMMAND nonlinear_interval_scoped_sqp_benchmark
    --phase-nodes 4,5,4,5
    --phase-nx 2,3,2,4
    --phase-nu 1,1,2,1
    --phase-parameters 1
    --interface-parameters 1
    --segment-parameters 1
    --segment-width 2
    --segment-stride 1
    --global-parameters 1
    --path-inequalities 2
    --collocation-degree 3
    --max-iterations 100
    --repeats 1
)
set_tests_properties(
  nonlinear_interval_scoped_ipm_regression
  PROPERTIES TIMEOUT 30
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
