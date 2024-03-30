//#include <SYCL/sycl.hpp>
#include <sycl/sycl.hpp>
#include<sys/sysinfo.h>
#include<sys/time.h>

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

int main()
{
    const int N=8192,M=8192,K=4096;

    unsigned long long int ClkPerSec;
    double NSecClk;
    unsigned long int start,end;
    double elapsed_count[10],Average = 0.0;

    queue q(default_selector_v);

    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    // Initialize Vectors and Print Values
    double *vector1 = static_cast<double*>(malloc(N*K*sizeof(double)));
    double *vector2 = static_cast<double*>(malloc(K*M*sizeof(double)));
    double *vector3 = static_cast<double*>(malloc(N*M*sizeof(double)));
    double *FNorm = static_cast<double*>(malloc_shared(sizeof(double),q));

    for(int i=0;i<N*K;i++)
        vector1[i] = 10.0;

    for(int i=0;i<K*M;i++)
        vector2[i] = 20.0;

    /*std::cout << "\nInput Vector1: \n";
    for(int i=0;i<(N*K);i+=K)
    {
        for(int j=i;j<(i+K);j++)
            std::printf("%.4f ",vector1[j]);
        
            //std::cout << vector1[j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nInput Vector2: \n";
    for(int i=0;i<(K*M);i+=M)
    {
        for(int j=i;j<(i+M);j++)
            std::printf("%.4f ",vector2[j]);
            //std::cout << vector2[j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
    */
    // Initialize Device USM and copy host data over to device for processing

    auto *vector1_device = static_cast<double*>(malloc_device<double>(N*K,q));
    auto *vector2_device = static_cast<double*>(malloc_device<double>(K*M,q));
    auto *vector3_device = static_cast<double*>(malloc_device<double>(N*M,q));

    Calibrate(&ClkPerSec,NSecClk);

    for(int count=0;count<10;count++)
    {
        FNorm[0] = 0.0;

        start = rdtsc();

        auto e1 = q.memcpy(vector1_device,vector1,(sizeof(double)*N*K));
        auto e2 = q.memcpy(vector2_device,vector2,(sizeof(double)*K*M));

        // Kernel to multiply the two Two-Dim Vectors

        q.parallel_for(range<2>(N,M), {e1,e2}, [=](auto index){

            int row = index.get_id(0);
            int col = index.get_id(1);

            double sum = 0.0;
            for(int k=0;k<K;k++)
                sum += vector1_device[row*K + k] * vector2_device[k*M + col];

            vector3_device[row*N + col] = sum;
            FNorm[0] += (sum*sum);
        }).wait();

        FNorm[0] = std::sqrt(FNorm[0]);

        auto e4 = q.parallel_for(range<2>(N,M), [=](auto index){

            int row = index.get_id(0);
            int col = index.get_id(1);

            vector3_device[row*N + col] = vector3_device[row*N + col]/FNorm[0];
        });

        q.memcpy(vector3,vector3_device,sizeof(double)*N*M,e4).wait();

        end = rdtsc();

        elapsed_count[count] = (double)(end - start)/ClkPerSec;
        std::printf("TTC : %.12f\n",elapsed_count[count]);
    }

    // Print updated Vector1 after Sum

    /*std::cout << "\nOutput Vector3: \n";
    for(int i=0;i<N*M;i+=M)
    {
        for(int j=i;j<(i+M);j++)
            std::printf("%.5f ",vector3_device[j]);
            //std::cout << vector3[j] << " ";
        std::cout << "\n";
    }*/

    for(int count=0;count<10;count++)
        Average += elapsed_count[count];

    Average = Average/10.0;
    printf("\nTime to compute Matrix Product (Copy + Computation + Copy) = %.12f\n",Average);

    free(vector1_device,q);
    free(vector2_device,q);
    free(vector3_device,q);
    free(vector1);
    free(vector2);
    free(vector3);

    return 0;
}
