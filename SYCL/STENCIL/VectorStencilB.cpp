#include <sycl/sycl.hpp>
#include<sys/sysinfo.h>
#include<sys/time.h>
//#include "tbb/tbb.h"

#define INDEX(N,i,j) (i*N + j)

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
    const int N=atoi(argv[3]),M=atoi(argv[4]);
    int index;
    unsigned long long int ClkPerSec;
    double NSecClk;
    unsigned long int start,end;
    double elapsed_count[10],Average = 0.0;


    queue cpu_selector(cpu_selector_v);
    queue gpu_selector(gpu_selector_v);
    queue q;
    if(strcmp(argv[2],"cpu") == 0)
        q = cpu_selector;
    else
        q = gpu_selector;

    //oneapi::tbb::task_group tg;
    //auto mp = tbb::global_control::max_allowed_parallelism;
    //oneapi::tbb::global_control gc(mp,atoi(argv[2]));


    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Max Compute Units : " << q.get_device().get_info<info::device::max_compute_units>() << std::endl;

    float *Mat_A       = static_cast<float*>(malloc_shared(N*M*sizeof(float),q));
    float *Mat_Stencil = static_cast<float*>(malloc_shared(N*M*sizeof(float),q));

    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++) 
            Mat_A[((i*N) + j)] = 1.0f;
            
    }

    /*std::cout << "Original Square Domain : \n";
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
            std::cout << H_a[(i*N) + j] << "\t";

        std::cout << "\n";
    }*/

    //auto *D_a = static_cast<float*>(malloc_device<float>(N*M,q));
    //auto *D_Stencil = static_cast<float*>(malloc_device<float>((N-2)*(M-2),q));

    Calibrate(&ClkPerSec,NSecClk);

    for(int count = 0;count < atoi(argv[1]);count++)
    {
        //if(FNorm[0] == 0.0f)
        //    FNorm[0] = 1.0f;
        //else
        //    FNorm[0] = 1.0f;
        //for(int i=0;i<M;i++)
        //{
        //    for(int j=0;j<N;j++)
        //        std::cout << H_a[(i*N) + j] << "\t";

        //    std::cout << "\n";
        //}

        start = rdtsc(); 

        // Kernel to compute the 5pt stencil and simultaneously the L2Norm
        q.parallel_for(range<2>(N-2,M-2), [=](auto index){
            int row = index.get_id(0) + 1;
            int col = index.get_id(1) + 1;

            Mat_Stencil[INDEX((N-2),(row-1),(col-1))] = (4*Mat_A[INDEX(N,row,col)] - Mat_A[INDEX(N,(row-1),col)] - Mat_A[INDEX(N,(row+1),col)] - Mat_A[INDEX(N,row,(col-1))] - Mat_A[INDEX(N,row,(col+1))]);
        }).wait();

        q.parallel_for(range<2>(N-2,M-2), [=](auto index){
            int row = index.get_id(0) + 1;
            int col = index.get_id(1) + 1;

            Mat_A[(row*N)+col] = (Mat_Stencil[INDEX((N-2),(row-1),(col-1))]);
        }).wait();

        end = rdtsc();
    
        elapsed_count[count] = (double)(end - start)/ClkPerSec;
        printf("TTC : %.12f\n",elapsed_count[count]);
    }

    // Print updated Vector1 after Sum

    //std::cout << "\nUpdated after 5pt Stencil : \n";
    //for(int i=0;i<N*M;i+=M)
    //{
    //    for(int j=i;j<(i+M);j++)
    //        std::cout << H_a[j] << "\t";
    //    std::cout << "\n";
    //}


    for(int count=1;count<(atoi(argv[1]));count++)
        Average += elapsed_count[count];
   
    std::cout << "\nTime to compute 5pt-Stencil + Power Method (Total) = " << Average << "\n"; 
    Average = Average/(atoi(argv[1]) - 1);
    std::cout << "\nTime to compute (Avg over " << atoi(argv[1]) << " loops) = " << Average << "\n";

    free(Mat_A,q);
    free(Mat_Stencil,q);
    //free(H_a);
    //free(FNorm,q);
}
