# Option to use system Google Test (default is OFF)
option(WITH_BUILD_GTEST "Use system-installed Google Test" OFF)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-report") # if GCC

if(NOT WITH_BUILD_GTEST)
  # Try to find system-installed Google Test
  find_package(GTest)
endif()

if(NOT WITH_BUILD_GTEST AND GTest_FOUND)
  message(STATUS "Using system-installed Google Test")
else()
  if(NOT WITH_BUILD_GTEST)
    message(STATUS "System Google Test not found. Falling back to fetching Google Test.")
  endif()

# Add Google Test
include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

# Add Google Test
include(GoogleTest)

# # Add Casadi
# message(STATUS "Looking for Casadi...")
find_package(casadi CONFIG REQUIRED)
include_directories(include ${CASADI_INCLUDE_DIRS}/casadi)
# message(STATUS "Casadi include dirs: ${CASADI_INCLUDE_DIRS}")

# Add Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Add pinocchio
# find_package(pinocchio REQUIRED)
# include_directories(${PINOCCHIO_INCLUDE_DIRS})
# message(STATUS "Pinocchio include dirs: ${PINOCCHIO_INCLUDE_DIRS}")

# Add the test executables
add_executable(vector_test linear_algebra/vector_test.cpp)
add_executable(matrix_test linear_algebra/matrix_test.cpp)
add_executable(lu_factorization_test linear_algebra/lu_factorization_test.cpp)
add_executable(jacobian_hessian_test ocp/jacobian_hessian_test.cpp)
add_executable(aug_system_solver_test ocp/aug_system_solver_test.cpp)
add_executable(problem_info_test ocp/problem_info_test.cpp)
add_executable(pd_system_solver_test ocp/pd_system_solver_test.cpp)
add_executable(pd_resto_solver_test ocp/pd_resto_solver_test.cpp)
add_executable(ocp_test ocp/ocp_test.cpp)
add_executable(nlp_ocp_test ocp/nlp_ocp_test.cpp)
add_executable(ocp_ip_data_solve_test ocp/ocp_ip_data_solve_test.cpp)
add_executable(ip_search_dir_test ocp/ip_search_dir_test.cpp)
add_executable(ip_search_dir_resto_test ocp/ip_search_dir_resto_test.cpp)
add_executable(ip_initializer_test ip_algorithm/ip_initializer_test.cpp)
add_executable(ip_filter_test ip_algorithm/ip_filter_test.cpp)
add_executable(options_test common/options_test.cpp)
add_executable(implicit_ocp_test implicit/implicit_ocp_test.cpp)
# add_executable(test_jacobian_decomposition implicit/test_jacobian_decomposition.cpp)
# add_executable(generate_models implicit/quadruped/cpp/generate_models.cpp implicit/quadruped/cpp/quadruped_helper_pinocchio.cpp)
# add_executable(casadi_test implicit/casadi_test.cpp)
add_executable(general_implicit_test implicit/general_implicit_test.cpp)
add_executable(general_lu_accelerated_test implicit/general_lu_accelerated_test.cpp)
add_executable(blasfeo_valgrind_test linear_algebra/blasfeo_valgrind_test.cpp)
add_executable(blasfeo_test linear_algebra/blasfeo_test.cpp)
add_executable(general_implicit_benchmark implicit/general_implicit_benchmark.cpp)
add_executable(test_lu_fact_complexity implicit/test_lu_fact_complexity.cpp)
add_executable(generate_visualization_files implicit/generate_visualization_files.cpp)
add_executable(test_block_LU_scaling reformulation/test_block_LU_scaling.cpp)
add_executable(benchmark implicit/general_implicit_benchmark_standalone.cpp)
add_executable(test_sparse_lu reformulation/test_sparse_lu.cpp)

# target_precompile_headers(implicit_ocp_test PRIVATE implicit/generators/pch.hpp)

# Link the test executables with Google Test and other required libraries
target_link_libraries(vector_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(options_test GTest::gtest GTest::gtest_main fatrop)
target_link_libraries(matrix_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(lu_factorization_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(jacobian_hessian_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(aug_system_solver_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(problem_info_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(pd_system_solver_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(pd_resto_solver_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(ocp_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(nlp_ocp_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(ocp_ip_data_solve_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(ip_search_dir_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(ip_search_dir_resto_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(ip_initializer_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(ip_filter_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(implicit_ocp_test blasfeo fatrop casadi::casadi)
# target_link_libraries(test_jacobian_decomposition blasfeo casadi fatrop)
# target_link_libraries(generate_models blasfeo fatrop pinocchio::pinocchio)
# target_link_libraries(casadi_test casadi::casadi)
target_link_libraries(general_implicit_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(general_lu_accelerated_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(blasfeo_valgrind_test GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(blasfeo_test blasfeo)
target_link_libraries(general_implicit_benchmark GTest::gtest GTest::gtest_main blasfeo fatrop)
target_link_libraries(test_lu_fact_complexity blasfeo fatrop)
target_link_libraries(generate_visualization_files blasfeo fatrop)
target_link_libraries(test_block_LU_scaling blasfeo fatrop)
target_link_libraries(benchmark blasfeo fatrop)
target_link_libraries(test_sparse_lu blasfeo fatrop casadi::casadi)

# Discover and add the tests
gtest_discover_tests(vector_test)
gtest_discover_tests(matrix_test)
gtest_discover_tests(lu_factorization_test)
gtest_discover_tests(jacobian_hessian_test)
gtest_discover_tests(aug_system_solver_test)
gtest_discover_tests(problem_info_test)
gtest_discover_tests(pd_system_solver_test)
gtest_discover_tests(pd_resto_solver_test)
gtest_discover_tests(ocp_test)
gtest_discover_tests(nlp_ocp_test)
gtest_discover_tests(ocp_ip_data_solve_test)
gtest_discover_tests(ip_search_dir_test)
gtest_discover_tests(ip_search_dir_resto_test)
gtest_discover_tests(ip_initializer_test)
gtest_discover_tests(ip_filter_test)
gtest_discover_tests(options_test)
gtest_discover_tests(general_implicit_test)
gtest_discover_tests(general_lu_accelerated_test)
gtest_discover_tests(blasfeo_valgrind_test)
gtest_discover_tests(general_implicit_benchmark)