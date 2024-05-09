//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>          //# sycl namespace
#include "oneapi/mkl/blas.hpp"  //# oneMKL DPC++ interface for BLAS functions
#include <sys/time.h>

// # The following project performs matrix multiplication using oneMKL / DPC++ with Unified Shared Memory (USM)
// # We will execute the simple operation A * B = C
// # The matrix B is set equal to the identity matrix such that A * B = A * I
// # After performing the computation, we will verify A * I = C -> A = C

using namespace sycl;
namespace mkl = oneapi::mkl;  //# shorten mkl namespace

unsigned long long rdtsc(void)
{
    unsigned long hi, lo;
    __asm__ __volatile__ ("xorl %%eax, %%eax \n  cpuid" ::: "%eax", "%ebx", "%ecx", "%edx");
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

static inline unsigned long long int GetTickCount()
{
#ifdef WIN32
    /* TODO find similar function on Windows */
#else
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return tp.tv_sec*1000+tp.tv_usec/1000;
}
#endif

void Calibrate(unsigned long long int *ClkPerSec,double NSecClk)
{
    unsigned long long int start,stop,diff;
    unsigned long long int starttick,stoptick,difftick;

    stoptick = GetTickCount();
    while(stoptick == (starttick=GetTickCount()));

    start = rdtsc();
    while((stoptick=GetTickCount())<(starttick+500));
    stop  = rdtsc();

    diff = stop-start;
    difftick = stoptick-starttick;

    *ClkPerSec = ( diff * (unsigned long long int)1000 )/ (unsigned long long int)(difftick);
    NSecClk = (double)1000000000 / (double)(__int64_t)*ClkPerSec;
}

int main(int argc, char *argv[]) {

    //# dimensions
    //int m = 3, n = 3, k = 3;
    
    //# scalar multipliers
    double alpha = 1.0, beta = 1.0;

    const int iteration_count = atoi(argv[1]);
    const int n = atoi(argv[3]);
    const int m = atoi(argv[4]);
    const int k = atoi(argv[5]);

    int ldA = m, ldB = k, ldC = m;

    queue cpu_queue(cpu_selector_v);
    queue gpu_queue(gpu_selector_v);
    queue q;
    if (strcmp(argv[2], "cpu") == 0)
        q = cpu_queue;
    else
        q = gpu_queue;

    unsigned long long int ClkPerSec;
    double NSecClk;
    unsigned long int start,end;
    double elapsed_count[10];
    double Average = 0.0;
    
    //# transpose status of matrices
    mkl::transpose transA = mkl::transpose::nontrans;
    mkl::transpose transB = mkl::transpose::nontrans;

    //### Step 1 - Create a queue with default selector.
    //queue q;
    device my_device = q.get_device();
    std::cout << "Device: " << my_device.get_info<info::device::name>() << "\n";

    //### Step 2 - Create a sycl event and allocate USM
    //# The later execution of the gemm operation is tied to this event
    //# The gemm operation will also make use of a vector of sycl events we can call 'gemm_dependencies'
    
    sycl::event gemm_done;
    std::vector<sycl::event> gemm_dependencies;
    
    //# Here, we allocate USM pointers for each matrix, using the special 'malloc_shared' function
    //# Make sure to template the function with the correct precision, and pass in our queue to the function call
    
    //float *A_usm = sycl::malloc_shared<float>(m * k, q);
    double *A_usm = static_cast<double*>(malloc_shared(m*k*sizeof(double),q));
    double *B_usm = static_cast<double*>(malloc_shared(k*n*sizeof(double),q));
    double *C_usm = static_cast<double*>(malloc_shared(m*n*sizeof(double),q));
    //float *B_usm = sycl::malloc_shared<float>(k * n, q);
    //float *C_usm = sycl::malloc_shared<float>(m * n, q);


/*    for(int i=0;i<m*k;i++)
        A_usm[i] = 10.0;

    for(int i=0;i<k*n;i++)
        B_usm[i] = 20.0;

    for (int i=0; i<m*n; i++)
        C_usm[i] = 0.0;*/

    //# define matrix A as the 3x3 matrix
    //# {{ 1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
    
    //### Step 3 - Execute gemm operation.
    //# Here, we fill in the familiar parameters for the gemm operation.
    //# However, we must also pass in the queue as the first parameter.
    //# We must also pass in our list of dependencies as the final parameter.
    //# We are also passing in our USM pointers as opposed to a buffer or raw data pointer.
    Calibrate(&ClkPerSec,NSecClk);

    for(int count=0;count<iteration_count;count++)
    {
        for(int i=0;i<m*k;i++)
            A_usm[i] = 10.0;

        for(int i=0;i<k*n;i++)
            B_usm[i] = 20.0;

        for (int i=0; i<m*n; i++)
            C_usm[i] = 0.0;
        //printf("Starting Loop!\n");
        start = rdtsc(); 
        gemm_done = mkl::blas::gemm(q, transA, transB, m, n, k, alpha, A_usm, ldA, B_usm, ldB, beta, C_usm, ldC, gemm_dependencies);
        //# We must now wait for the given event to finish before accessing any data involved in the operation
        //# Otherwise, we may access data before the operation has completed, or before it has been returned to the host
        gemm_done.wait();
	end = rdtsc();
        elapsed_count[count] = (double)(end-start)/ClkPerSec;
        printf("TTC : %0.12f\n",elapsed_count[count]);
    }

    for(int count=1;count<iteration_count;count++)
        Average += elapsed_count[count];

    Average = Average/(iteration_count - 1);
    printf("\nTime to compute Matrix Product = %0.12f \n",Average);

    //int status = 0;

    //# verify C matrix using USM data
    //std::cout << "\n";
    //std::cout << "C = \n";
    //for (int i = 0; i < m; ++i) {
    //    for (int j = 0; j < n; ++j) {
    //        if (A_usm[i*m+j] != C_usm[i*m+j]) status = 1;
    //        std::cout << C_usm[i*m+j] << " ";
    //    }
    //    std::cout << "\n";
    //}
    //std::cout << "\n";

    //# free usm pointers
    sycl::free(A_usm, q);
    sycl::free(B_usm, q);
    sycl::free(C_usm, q);

    //status == 0 ? std::cout << "Verified: A = C\n" : std::cout << "Failed: A != C\n";
    return 0;
}
