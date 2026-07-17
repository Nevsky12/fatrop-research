#include "fatrop/ocp/problem_info.hpp"
#include "gtest/gtest.h"

namespace fatrop
{
    namespace test
    {

class ProblemInfoTest : public ::testing::Test
{
protected:
    ProblemDims createSampleDims()
    {
        return ProblemDims(
            3, // 3 stages
            std::vector<Index>{2, 3, 4}, // number_of_controls
            std::vector<Index>{4, 5, 6}, // number_of_states
            std::vector<Index>{1, 2, 3}, // number_of_eq_constraints
            std::vector<Index>{6, 7, 8}  // number_of_ineq_constraints
        );
    }
};

TEST_F(ProblemInfoTest, ConstructorAndOffsets)
{
    ProblemDims dims = createSampleDims();
            ProblemInfo problem_info(dims);

            // Test dimensions
            EXPECT_EQ(problem_info.dims.K, 3);

            // Test primal offsets
            EXPECT_EQ(problem_info.offset_primal, 0);
            EXPECT_EQ(problem_info.offsets_primal_u, std::vector<Index>({0, 6, 14}));
            EXPECT_EQ(problem_info.offsets_primal_x, std::vector<Index>({2, 9, 18}));

            // // Test slack offsets
            EXPECT_EQ(problem_info.offset_slack, 24);
            EXPECT_EQ(problem_info.offsets_slack, std::vector<Index>({0, 6, 13}));

            // // Test equality constraint numbers
            EXPECT_EQ(problem_info.number_of_g_eq_dyn, 11);
            EXPECT_EQ(problem_info.number_of_g_eq_path, 6);
            EXPECT_EQ(problem_info.number_of_g_eq_slack, 21);

            // // Test equality constraint offsets
            EXPECT_EQ(problem_info.offset_g_eq_dyn, 6);
            EXPECT_EQ(problem_info.offset_g_eq_path, 0);
            EXPECT_EQ(problem_info.offset_g_eq_slack, 17);

            EXPECT_EQ(problem_info.offsets_g_eq_path, std::vector<Index>({0, 1, 3}));
            EXPECT_EQ(problem_info.offsets_g_eq_dyn, std::vector<Index>({6, 11, 0}));
            EXPECT_EQ(problem_info.offsets_g_eq_slack, std::vector<Index>({17, 23, 30}));
        }

TEST_F(ProblemInfoTest, AppendsOneCopyGlobalParametersAfterTrajectory)
{
    ProblemDims dims(
        3,
        std::vector<Index>{2, 3, 4},
        std::vector<Index>{4, 5, 6},
        std::vector<Index>{1, 2, 3},
        std::vector<Index>{6, 7, 8},
        5);
    ProblemInfo const info(dims);

    EXPECT_EQ(info.number_of_trajectory_variables, 24);
    EXPECT_EQ(info.number_of_global_parameters, 5);
    EXPECT_EQ(info.offset_primal_global, 24);
    EXPECT_EQ(info.number_of_primal_variables, 29);
    EXPECT_EQ(info.offset_slack, 29);
    EXPECT_EQ(info.pd_orig_offset_slack, 29);
    EXPECT_EQ(info.pd_resto_offset_slack, 29);
    EXPECT_EQ(info.offsets_primal_u, std::vector<Index>({0, 6, 14}));
    EXPECT_EQ(info.offsets_primal_x, std::vector<Index>({2, 9, 18}));
}

TEST_F(ProblemInfoTest, StageBorderDimensionDoesNotAllocatePrimalCopies)
{
    ProblemDims dims(
        2,
        std::vector<Index>{0, 0},
        std::vector<Index>{1, 1},
        std::vector<Index>{3, 0},
        std::vector<Index>{0, 0},
        0,
        std::vector<Index>{2, 0});
    ProblemInfo const info(dims);

    EXPECT_EQ(
        info.dims.number_of_stage_border_variables,
        std::vector<Index>({2, 0}));
    EXPECT_EQ(info.number_of_trajectory_variables, 2);
    EXPECT_EQ(info.number_of_global_parameters, 0);
    EXPECT_EQ(info.number_of_primal_variables, 2);
    EXPECT_EQ(info.number_of_g_eq_path, 3);
}

    } // namespace test
} // namespace fatrop
