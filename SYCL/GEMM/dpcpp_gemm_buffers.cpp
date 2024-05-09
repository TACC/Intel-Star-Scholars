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

//# The following project performs matrix multiplication using oneMKL / DPC++ with buffers.
//# We will execute the simple operation A * B = C
//# The matrix B is set equal to the identity matrix such that A * B = A * I
//# After performing the computation, we will verify A * I = C -> A = C

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
    
    //int m = 8192, n = 8192, k = 8192;
    
    //# leading dimensions
    
    
    //# scalar multipliers
    
    float alpha = 1.0, beta = 1.0;

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
    
    //# matrix data
    
    std::vector<double> A(m*k, 20.0); // = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<double> B(k*n, 10.0); // = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> C(m*n, 0.0); // = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    //### Step 1 - Create a queue with default selector.
    
    //queue q(cpu_selector_v);
    device my_device = q.get_device();
    std::cout << "Device: " << my_device.get_info<info::device::name>() << "\n";

    //### Step 2 - Create buffers to hold our matrix data.
    //# Buffer objects can be constructed given a container
    //# Observe the creation of buffers for matrices A and B.
    //# Try and create a third buffer for matrix C called C_buffer.
    //# The solution is shown in the hidden cell below.
    
    buffer A_buffer(A);
    buffer B_buffer(B);
    /* define C_buffer here */
    buffer C_buffer(C);
    
    //### Step 3 - Execute gemm operation.
    //# Here, we need only pass in our queue and other familiar matrix multiplication parameters.
    //# This includes the dimensions and data buffers for matrices A, B, and C.
    Calibrate(&ClkPerSec,NSecClk);


    for(int count=0;count<iteration_count;count++)
    {
        start = rdtsc();    
        mkl::blas::gemm(q, transA, transB, m, n, k, alpha, A_buffer, ldA, B_buffer, ldB, beta, C_buffer, ldC);
        host_accessor C_acc(C_buffer, read_only);
	end = rdtsc();
        elapsed_count[count] = (double)(end-start)/ClkPerSec;
        printf("TTC : %f\n",elapsed_count[count]);
    }


    //### Step 6 - Observe creation of accessors to retrieve data from A_buffer and C_buffer.
    
    host_accessor A_acc(A_buffer, read_only);
    host_accessor C_acc(C_buffer, read_only);

    for(int count=1;count<iteration_count;count++)
        Average += elapsed_count[count];

    Average = Average/(iteration_count-1);
    printf("\nTime to compute Matrix Product = %0.12f \n",Average);

    //int status = 0;

    // verify C matrix using accessor to observe values held in C_buffer
    
    std::cout << std::endl;
    //std::cout << "C = " << std::endl;
    //for (int i = 0; i < m; ++i) {
    //    for (int j = 0; j < n; ++j) {
    //        if (A_acc[i*m+j] != C_acc[i*m+j]) status = 1;
            //std::cout << C_acc[i*m+j] << " ";
    //    }
        //std::cout << std::endl;
    //}
    //std::cout << std::endl;

    //status == 0 ? std::cout << "Verified: A = C" << std::endl : std::cout << "Failed: A != C" << std::endl;
    return 0;
}
