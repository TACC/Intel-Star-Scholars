#include <sycl/sycl.hpp>
#include<sys/sysinfo.h>
#include<sys/time.h>
#include "tbb/tbb.h"

//using namespace hipsycl::sycl;
using namespace sycl;

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

int main(int argc,char *argv[])
{
    const int N=40000,M=40000;
    int index;
    unsigned long long int ClkPerSec;
    double NSecClk;
    unsigned long int start,end;
    double elapsed_count[10],Average = 0.0;

    queue q(gpu_selector_v);
    //oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, atoi(argv[1]));
    

    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Max Compute Units : " << q.get_device().get_info<info::device::max_compute_units>() << std::endl;

    // Initialise Bordered-Array.
    float *H_a = static_cast<float*>(malloc(N*M*sizeof(float)));
    float *FNorm = static_cast<float*>(malloc_shared(sizeof(float),q));
    FNorm[0] = 0.0f;

    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++) 
            H_a[((i*N) + j)] = 1.0f;
            
    }

    /*std::cout << "Original Square Domain : \n";
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
            std::cout << H_a[(i*N) + j] << "\t";

        std::cout << "\n";
    }*/

    auto *D_a = static_cast<float*>(malloc_device<float>(N*M,q));
    auto *D_Stencil = static_cast<float*>(malloc_device<float>((N-2)*(M-2),q));

    Calibrate(&ClkPerSec,NSecClk);

    for(int count = 0;count < atoi(argv[1]);count++)
    {
        if(FNorm[0] == 0.0f)
            FNorm[0] = 1.0f;
        else
            FNorm[0] = 0.0f;
        /*for(int i=0;i<M;i++)
        {
            for(int j=0;j<N;j++)
                std::cout << H_a[(i*N) + j] << "\t";

            std::cout << "\n";
        }*/

        start = rdtsc(); 

        q.memcpy(D_a,H_a,(sizeof(float)*N*M)).wait();

        // Kernel to compute the 5pt stencil and simultaneously the L2Norm
        q.parallel_for(range<2>(N-2,M-2), [=](auto index){
            int row = index.get_id(0) + 1;
            int col = index.get_id(1) + 1;

            D_Stencil[((row-1)*(N-2)) + (col-1)] = (4*D_a[((row*N) + col)] - D_a[((row-1)*N) + col] - D_a[((row+1)*N) + col] - D_a[((row)*N) + (col-1)] - D_a[((row)*N) + (col+1)]);
            FNorm[0] += (D_Stencil[((row-1)*(N-2)) + (col-1)] * D_Stencil[((row-1)*(N-2)) + (col-1)]);
        });

        //printf("Frobenius Norm : %.12f\n",FNorm[0]);
	FNorm[0] = std::sqrt(FNorm[0]);
        //printf("Frobenius Norm : %.12f\n",FNorm[0]);

        // Kernel to scale the interior by (Stencil Updated Value / L2Norm)
	q.parallel_for(range<2>(N-2,M-2), [=](auto index){
            int row = index.get_id(0) + 1;
            int col = index.get_id(1) + 1;

            D_a[(row*N)+col] = (D_Stencil[((row-1)*(N-2)) + (col-1)]/FNorm[0]);
        });

        q.memcpy(H_a,D_a,sizeof(float)*N*M).wait();
    
        end = rdtsc();
    
        elapsed_count[count] = (double)(end - start)/ClkPerSec;
        //printf("TTC : %.12f\n",elapsed_count[count]);
    }

    // Print updated Vector1 after Sum

    /*std::cout << "\nUpdated after 5pt Stencil : \n";
    for(int i=0;i<N*M;i+=M)
    {
        for(int j=i;j<(i+M);j++)
            std::cout << H_a[j] << "\t";
        std::cout << "\n";
    }*/


    for(int count=0;count<(atoi(argv[1]));count++)
        Average += elapsed_count[count];
   
    std::cout << "\nTime to compute 5pt-Stencil + Power Method (Total) = " << Average << "\n"; 
    Average = Average/(atoi(argv[1]));
    std::cout << "\nTime to compute (Avg over " << atoi(argv[1]) << " loops) = " << Average << "\n";

    free(D_a,q);
    free(D_Stencil,q);
    free(H_a);
    free(FNorm,q);
}
