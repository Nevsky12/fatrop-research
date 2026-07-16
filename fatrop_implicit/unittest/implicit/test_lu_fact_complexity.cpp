#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/linear_algebra/lu_factorization.hpp"

#include <chrono>
#include <fstream>
#include <optional>

using namespace fatrop;

int main(){
    int nb_runs = 10000;

    std::ofstream f("lu_fact_complexity.csv");
    f << "m,n,time_ns\n";
    
    std::vector<int> m(nb_runs);
    std::vector<int> n(nb_runs);
    std::vector<MatRealAllocated> A;
    std::vector<PermutationMatrix> Pl;
    std::vector<PermutationMatrix> Pr;

    A.reserve(nb_runs);
    Pl.reserve(nb_runs);
    Pr.reserve(nb_runs);
    for (int k = 0; k < nb_runs; k++){
        m[k] = 1 + rand() % 100;
        n[k] = 1 + rand() % 100;

        A.emplace_back(m[k],n[k]);
        for (int i = 0; i < m[k]; i++){
            for (int j = 0; j < n[k]; j++){
                A[k](i,j) = rand() / double(RAND_MAX);
            }
        }
        Pl.emplace_back(m[k]);
        Pr.emplace_back(n[k]);
    }
    std::cout << "allocations done" << std::endl;

    long int duration_ns;
    for (int k = 0; k < nb_runs; k++){        
        Index rank;
        std::cout << "Run " << k+1 << "/" << nb_runs << ": m=" << m[k] << ", n=" << n[k] << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        fatrop_lu_fact_transposed(n[k], m[k], m[k], rank, &A[k].mat(), Pl[k], Pr[k], 1e-5);
        auto end = std::chrono::high_resolution_clock::now();
        duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        f << m[k] << "," << n[k] << "," << duration_ns << "\n";
    }
    std::cout << "here" << std::endl;
}