#include <iostream>
#include <cassert>
#include <random>
#include <blasfeo.h>

void fill_randomly(int m, int n, blasfeo_dmat *sM){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            double value = static_cast<double>(rand()) / RAND_MAX;
            blasfeo_dgein1(value, sM, i, j);
        }
    }
}

int main(int argc, char** argv){
    srand(0);

    int m = 14;
    int n = 14;
    int k = 5;

    // check if args contain "m=..." and "n=..."
    for (int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if (arg.find("m=") == 0){
            m = std::stoi(arg.substr(2));
        } else if (arg.find("n=") == 0){
            n = std::stoi(arg.substr(2));
        } else if (arg.find("k=") == 0){
            k = std::stoi(arg.substr(2));
        }
    }

    int m_padded = m;
    int n_padded = n;
    int k_padded = k;
    bool ceil_4 = false;
    bool ceil_8 = false;
    if (ceil_4){
        m_padded += 4 - (m % 4);
        n_padded += 4 - (n % 4);
        k_padded += 4 - (k % 4);
    }
    if (ceil_8){
        m_padded += 8 - (m % 8);
        n_padded += 8 - (n % 8);
        k_padded += 8 - (k % 8);
    }
    std::cout << "m: " << m << ", n: " << n << ", k: " << k << std::endl;
    std::cout << "m_padded: " << m_padded << ", n_padded: " << n_padded << ", k_padded: " << k_padded << std::endl;

    struct blasfeo_dmat sA, sB, sAt, sBt, sM_ref, sM1, sM2, sM3, sM4;
    blasfeo_allocate_dmat(k_padded, m_padded, &sA);
    blasfeo_allocate_dmat(n_padded, k_padded, &sB);
    blasfeo_allocate_dmat(m_padded, k_padded, &sAt);
    blasfeo_allocate_dmat(k_padded, n_padded, &sBt);
    blasfeo_allocate_dmat(m_padded, n_padded, &sM_ref);
    blasfeo_allocate_dmat(m_padded, n_padded, &sM1);
    blasfeo_allocate_dmat(m_padded, n_padded, &sM2);
    blasfeo_allocate_dmat(m_padded, n_padded, &sM3);
    blasfeo_allocate_dmat(m_padded, n_padded, &sM4);

    blasfeo_dgese(k_padded, m_padded, 0, &sA, 0, 0);
    blasfeo_dgese(n_padded, k_padded, 0, &sB, 0, 0);
    blasfeo_dgese(m_padded, k_padded, 0, &sAt, 0, 0);
    blasfeo_dgese(k_padded, n_padded, 0, &sBt, 0, 0);
    blasfeo_dgese(m_padded, n_padded, 0, &sM_ref, 0, 0);
    blasfeo_dgese(m_padded, n_padded, 0, &sM1, 0, 0);
    blasfeo_dgese(m_padded, n_padded, 0, &sM2, 0, 0);
    blasfeo_dgese(m_padded, n_padded, 0, &sM3, 0, 0);
    blasfeo_dgese(m_padded, n_padded, 0, &sM4, 0, 0);

    // fill randomly
    fill_randomly(k, m, &sA);
    fill_randomly(n, k, &sB);

    // construct transposes
    blasfeo_dgetr(k, m, &sA, 0, 0, &sAt, 0, 0);
    blasfeo_dgetr(n, k, &sB, 0, 0, &sBt, 0, 0);    

    // manual multiplication
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            double sum = 0.0;
            for (int p = 0; p < k; ++p){
                double sa = blasfeo_dgeex1(&sA, p, i);
                double sb = blasfeo_dgeex1(&sB, j, p);
                sum += sa * sb;
            }
            blasfeo_dgein1(sum, &sM_ref, i, j);
        }
    }

    // blasfeo calls
    blasfeo_dgemm_tt(m, n, k, 1.0, &sA, 0, 0, &sB, 0, 0, 1.0, &sM1, 0, 0, &sM1, 0, 0);
    blasfeo_dgemm_nt(m, n, k, 1.0, &sAt, 0, 0, &sB, 0, 0, 1.0, &sM2, 0, 0, &sM2, 0, 0);
    blasfeo_dgemm_nn(m, n, k, 1.0, &sAt, 0, 0, &sBt, 0, 0, 1.0, &sM3, 0, 0, &sM3, 0, 0);
    blasfeo_dgemm_tn(m, n, k, 1.0, &sA, 0, 0, &sBt, 0, 0, 1.0, &sM4, 0, 0, &sM4, 0, 0);

    // compare results
    std::cout << "reference:\n" << std::endl;
    blasfeo_print_dmat(m, n, &sM_ref, 0, 0);
    std::cout << "blasfeo_calls:\n" << std::endl;
    std::cout << "blasfeo call 1 (tt):\n" << std::endl; blasfeo_print_dmat(m, n, &sM1, 0, 0);
    std::cout << "blasfeo call 2 (nt):\n" << std::endl; blasfeo_print_dmat(m, n, &sM2, 0, 0);
    std::cout << "blasfeo call 3 (nn):\n" << std::endl; blasfeo_print_dmat(m, n, &sM3, 0, 0);
    std::cout << "blasfeo call 4 (tn):\n" << std::endl; blasfeo_print_dmat(m, n, &sM4, 0, 0);

    // compute error norms
    double err1, err2, err3, err4;
    err1 = err2 = err3 = err4 = 0.0;

    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            double ref = blasfeo_dgeex1(&sM_ref, i, j);
            double val1 = blasfeo_dgeex1(&sM1, i, j);
            double val2 = blasfeo_dgeex1(&sM2, i, j);
            double val3 = blasfeo_dgeex1(&sM3, i, j);
            double val4 = blasfeo_dgeex1(&sM4, i, j);
            err1 += (ref - val1) * (ref - val1);
            err2 += (ref - val2) * (ref - val2);
            err3 += (ref - val3) * (ref - val3);
            err4 += (ref - val4) * (ref - val4);
        }
    }

    err1 = sqrt(err1);
    err2 = sqrt(err2);
    err3 = sqrt(err3);
    err4 = sqrt(err4);

    std::cout << "Error norms:\n";
    std::cout << "Error norm for blasfeo call 1 (tt): " << err1 << std::endl;
    std::cout << "Error norm for blasfeo call 2 (nt): " << err2 << std::endl;
    std::cout << "Error norm for blasfeo call 3 (nn): " << err3 << std::endl;
    std::cout << "Error norm for blasfeo call 4 (tn): " << err4 << std::endl;

    blasfeo_free_dmat(&sA);
    blasfeo_free_dmat(&sB);
    blasfeo_free_dmat(&sAt);
    blasfeo_free_dmat(&sBt);
    blasfeo_free_dmat(&sM_ref);
    blasfeo_free_dmat(&sM1);
    blasfeo_free_dmat(&sM2);
    blasfeo_free_dmat(&sM3);
    blasfeo_free_dmat(&sM4);

    return 0;
}