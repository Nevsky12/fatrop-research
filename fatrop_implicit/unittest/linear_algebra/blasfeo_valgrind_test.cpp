#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/linear_algebra/blasfeo_operations.hpp"
#include "../random_matrix.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <cassert>
#include <random>

void perform_dgemm_nn(int m, int n, int k, int row_offset, bool ceil_4, bool ceil_8, bool block){
    std::string name = "\n\nComputing blasfeo_dgemm_nn";
    int nb_rows = m + row_offset;
    int nb_cols = k;
    if (ceil_4){ nb_rows += 4;} else if (ceil_8){ nb_rows += 8;}
    if (ceil_4){ nb_cols += 4;} else if (ceil_8){ nb_cols += 8;}
    name += " (" + std::to_string(nb_rows) + ")";
    name += " row offset = " + std::to_string(row_offset);

    fatrop::MatRealAllocated A = fatrop::test::random_matrix(nb_rows, nb_cols);
    fatrop::MatRealAllocated B(k, n);
    fatrop::MatRealAllocated C(m, n);
    fatrop::MatRealAllocated D(m, n);

    std::cout << name << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    if (block){
        blasfeo_dgemm_nn(m, n, k, 1.0, &A.block(m, k, row_offset, 0).mat(), 0, 0, &B.mat(), 0, 0, 1.0, &C.mat(), 0, 0, &D.mat(), 0, 0);
    } else {
        blasfeo_dgemm_nn(m, n, k, 1.0, &A.mat(), row_offset, 0, &B.mat(), 0, 0, 1.0, &C.mat(), 0, 0, &D.mat(), 0, 0);
    }
    std::cout << "Done" << std::endl;
}

void perform_dgemm_nt(int m, int n, int k, int row_offset, bool ceil_4, bool ceil_8, bool block){
    std::string name = "\nComputing blasfeo_dgemm_nt";
    int nb_rows = m + row_offset;
    int nb_cols = k;
    if (ceil_4){ nb_rows += 4;} else if (ceil_8){ nb_rows += 8;}
    if (ceil_4){ nb_cols += 4;} else if (ceil_8){ nb_cols += 8;}
    name += " (" + std::to_string(nb_rows) + ")";
    name += " row offset = " + std::to_string(row_offset);

    fatrop::MatRealAllocated A = fatrop::test::random_matrix(nb_rows, nb_cols);
    fatrop::MatRealAllocated B(n, k);
    fatrop::MatRealAllocated C(m, n);
    fatrop::MatRealAllocated D(m, n);

    std::cout << name << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    if (block){
        blasfeo_dgemm_nt(m, n, k, 1.0, &A.block(m, k, row_offset, 0).mat(), 0, 0, &B.mat(), 0, 0, 1.0, &C.mat(), 0, 0, &D.mat(), 0, 0);
    } else {
        blasfeo_dgemm_nt(m, n, k, 1.0, &A.mat(), row_offset, 0, &B.mat(), 0, 0, 1.0, &C.mat(), 0, 0, &D.mat(), 0, 0);
    }
    std::cout << "Done" << std::endl;
}

int main(){
    int m = 3;
    int n = 2;
    int k = 5;
    int row_offset = 3;

    // randomize dimensions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 3);
    m = dis(gen);
    n = dis(gen);
    k = dis(gen);
    row_offset = dis(gen);

    std::cout << "Without row offset" << std::endl;
    perform_dgemm_nn(m, n, k, 0, false, false, false);

    std::cout << "With row offset" << std::endl;
    perform_dgemm_nn(m, n, k, row_offset, false, false, false);
    perform_dgemm_nn(m, n, k, row_offset, true, false, false);
    perform_dgemm_nn(m, n, k, row_offset, false, true, false);

    std::cout << "With blocking" << std::endl;
    perform_dgemm_nn(m, n, k, row_offset, false, false, true);

    
    std::cout << "Without row offset" << std::endl;
    perform_dgemm_nt(m, n, k, 0, false, false, false);

    std::cout << "With row offset" << std::endl;
    perform_dgemm_nt(m, n, k, row_offset, false, false, false);
    perform_dgemm_nt(m, n, k, row_offset, true, false, false);
    perform_dgemm_nt(m, n, k, row_offset, false, true, false);

    std::cout << "With blocking" << std::endl;
    perform_dgemm_nt(m, n, k, row_offset, false, false, true);


    return 0;
}