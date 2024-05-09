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
    
    std::vector<double> vector1(n*m, 10.0);
    std::vector<double> vector2(n*m, 20.0);
    buffer vector1_buf(vector1);
    buffer vector2_buf(vector2);

    //queue q(cpu_selector_v);
    device my_device = q.get_device();
    std::cout << "Device: " << my_device.get_info<info::device::name>() << "\n";

    Calibrate(&ClkPerSec,NSecClk);
    int64_t incx = 1;
    int64_t incy = 1;
    double alpha = 1.5;
 
    for(int count=0;count<iteration_count;count++)
    {
        start = rdtsc(); 
	mkl::blas::axpy(q, n*m, alpha, vector1_buf, incx, vector2_buf, incy);
	host_accessor vector2_acc(vector2_buf, read_only);
	end = rdtsc();
        elapsed_count[count] = (double)(end-start)/ClkPerSec;
        //printf("TTC : %f\n",elapsed_count[count]);
    }

    for(int count=1;count<iteration_count;count++)
        Average += elapsed_count[count];

    Average = Average/(iteration_count-1);
    printf("\nTime to compute Matrix Product = %0.12f \n",Average);

    std::cout << std::endl;
    return 0;
}
