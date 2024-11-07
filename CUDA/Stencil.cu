#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <sys/sysinfo.h>
#include <sys/time.h>

#define INDEX(N,i,j) (i*N + j)

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

// Kernel to compute the 5-point stencil and accumulate the norm
__global__ void stencil_kernel(const float *Mat_A, float *Mat_Stencil, int N, int M, float *FNorm) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (row < N-1 && col < M-1) {
        float stencil_value = 4.0f * Mat_A[row * M + col]
                             - Mat_A[(row - 1) * M + col]
                             - Mat_A[(row + 1) * M + col]
                             - Mat_A[row * M + (col - 1)]
                             - Mat_A[row * M + (col + 1)];
        Mat_Stencil[(row - 1) * (M - 2) + (col - 1)] = stencil_value;
        atomicAdd(FNorm, stencil_value * stencil_value);
    }
}



