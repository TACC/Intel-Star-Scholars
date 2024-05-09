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

    //# scalar multipliers
    double alpha = 1.5;

    const int iteration_count = atoi(argv[1]);
    const int n = atoi(argv[3]);
    const int m = atoi(argv[4]);

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
    double elapsed_count[500];
    double Average = 0.0;
    
    device my_device = q.get_device();
    std::cout << "Device: " << my_device.get_info<info::device::name>() << "\n";

    //### Step 2 - Create a sycl event and allocate USM
    sycl::event axpy_done;
    std::vector<sycl::event> axpy_dependencies;
    
    //# Here, we allocate USM pointers for each matrix, using the special 'malloc_shared' function
    //# Make sure to template the function with the correct precision, and pass in our queue to the function call
    double *vector1 = static_cast<double*>(malloc(n*m*sizeof(double)));
    double *vector2 = static_cast<double*>(malloc(n*m*sizeof(double)));

    for(int i=0;i<n*m;i++)
        vector1[i] = 10.0;

    for(int i=0;i<n*m;i++)
        vector2[i] = 20.0;

    auto *vector1_usm = static_cast<double*>(malloc_device<double>(n*m,q));
    auto *vector2_usm = static_cast<double*>(malloc_device<double>(n*m,q));

    Calibrate(&ClkPerSec,NSecClk);

    int64_t incx = 1;
    int64_t incy = 1;
    for(int count=0;count<iteration_count;count++)
    {
        //printf("Starting Loop!\n");
        start = rdtsc();

	auto e1 = q.memcpy(vector1_usm,vector1,(sizeof(double)*n*m));
        auto e2 = q.memcpy(vector2_usm,vector2,(sizeof(double)*n*m));

        e1.wait();
        e2.wait();

        //start = rdtsc();
	axpy_done = mkl::blas::axpy(q, n*m, alpha, vector1_usm, incx, vector2_usm, incy, axpy_dependencies);
        //# We must now wait for the given event to finish before accessing any data involved in the operation
        //# Otherwise, we may access data before the operation has completed, or before it has been returned to the host
        axpy_done.wait();
	end = rdtsc();
        elapsed_count[count] = (double)(end-start)/ClkPerSec;
        //printf("TTC : %0.12f\n",elapsed_count[count]);
    }

    for(int count=1;count<iteration_count;count++)
        Average += elapsed_count[count];

    Average = Average/(iteration_count - 1);
    printf("\nTime to compute Matrix Product = %0.12f \n",Average);

    //# free usm pointers
    free(vector1);
    free(vector2);
    sycl::free(vector1_usm, q);
    sycl::free(vector2_usm, q);
    return 0;
}
