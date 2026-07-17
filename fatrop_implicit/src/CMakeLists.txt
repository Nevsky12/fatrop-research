# Define source files for the fatrop library
set(FATROP_SOURCES
    linear_algebra/lu_factorization.cpp
    linear_algebra/vector.cpp
    ocp/aug_system_solver.cpp
    ocp/dims.cpp
    ocp/jacobian.cpp
    ocp/hessian.cpp
    ocp/interval_scoped_kkt_solver.cpp
    ocp/ip_alg_builder.cpp
    ocp/ip_algorithm.cpp
    ocp/ip_convergence_check.cpp
    ocp/ip_convergence_check_resto.cpp
    ocp/ip_data.cpp
    ocp/ip_eq_mult_initializer.cpp
    ocp/ip_initializer.cpp
    ocp/ip_initializer_resto.cpp
    ocp/ip_iterate.cpp
    ocp/ip_iteration_output.cpp
    ocp/ip_iteration_output_resto.cpp
    ocp/ip_linesearch.cpp
    ocp/ip_nlp_orig.cpp
    ocp/ip_nlp_resto.cpp
    ocp/ip_resto_phase_min_cl1.cpp
    ocp/ip_search_dir.cpp
    ocp/nlp_ocp.cpp
    ocp/parametric_implicit_nlp_ocp.cpp
    ocp/pd_system_orig.cpp
    ocp/pd_system_resto.cpp
    ocp/pd_solver_orig.cpp
    ocp/pd_solver_resto.cpp
    ocp/problem_info.cpp
    ip_algorithm/ip_filter.cpp
    ip_algorithm/ip_mu_update.cpp
    ip_algorithm/ip_utils.cpp
)

# Create the fatrop shared library
add_library(fatrop SHARED ${FATROP_SOURCES})

# Set properties for the fatrop target
set_target_properties(fatrop PROPERTIES
    VERSION ${PROJECT_VERSION}
)

# Link fatrop with blasfeo
target_link_libraries(fatrop PUBLIC blasfeo)

# Specify include directories for the target
target_include_directories(fatrop
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


# Define install targets
set(INSTALL_TARGETS fatrop)

# Export the target
install(TARGETS ${INSTALL_TARGETS}
    EXPORT fatropTargets
    LIBRARY DESTINATION ${LIBRARY_INSTALL_DIR}
    ARCHIVE DESTINATION ${ARCHIVE_INSTALL_DIR}
    RUNTIME DESTINATION ${RUNTIME_INSTALL_DIR}
    INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
)

# Install header files (only .hpp)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/fatrop
        DESTINATION ${INCLUDE_INSTALL_DIR}
        FILES_MATCHING PATTERN "*.hpp")

# Install a relocatable package.  The target's public interface contains
# BLASFEO, so consumers must import that target before the FATROP targets file.
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/fatropConfig.cmake.in
    ${PROJECT_BINARY_DIR}/fatropConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
)
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/fatropConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(EXPORT fatropTargets
        FILE fatropTargets.cmake
        NAMESPACE fatrop::
        DESTINATION ${CMAKE_INSTALL_DIR})
install(FILES
        ${PROJECT_BINARY_DIR}/fatropConfig.cmake
        ${PROJECT_BINARY_DIR}/fatropConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DIR})
