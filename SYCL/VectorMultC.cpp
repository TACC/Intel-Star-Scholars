//#include <SYCL/sycl.hpp>
#include<stdio.h>
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

    // Using time point and system_clock
    unsigned long long int ClkPerSec;
    double NSecClk;
    unsigned long int start,end;
    double elapsed_count[10],Average = 0.0;

    queue q(default_selector_v);

    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    // Initialize Vectors and Print Values
    std::vector<double> vector1(N*K,10.0);
    std::vector<double> vector2(K*M,20.0);
    std::vector<double> vector3(N*M,0.0);
    double *FNorm = static_cast<double*>(malloc_shared(sizeof(double),q));

    /*std::cout << "\nInput Vector1: \n";
    for(int i=0;i<(N*K);i+=K)
    {
        for(int j=i;j<(i+K);j++)
            std::cout << vector1[j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nInput Vector2: \n";
    for(int i=0;i<(K*M);i+=M)
    {
        for(int j=i;j<(i+M);j++)
            std::cout << vector2[j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
    */

    buffer<double,2> vector1_device(vector1.data(),range<2>(N,K));
    buffer<double,2> vector2_device(vector2.data(),range<2>(K,M));
    buffer<double,2> vector3_device(vector3.data(),range<2>(N,M));

    // Shared Unified Memory created, without the need for copy
    Calibrate(&ClkPerSec,NSecClk);
    
    for(int count=0;count<10;count++)
    {
        FNorm[0] = 0.0;

        start = rdtsc();

        // Kernel to add the two Two-Dim Vectors
        q.submit([&] (handler &h) 
        {
            accessor buf1(vector1_device,h);
            accessor buf2(vector2_device,h);
            accessor buf3(vector3_device,h);

            h.parallel_for(range<2>(N,M), [=](auto index){
                int row = index.get_id(0);
                int col = index.get_id(1);

                double sum = 0.0;
                for(int k=0;k<K;k++)
                    sum += buf1[row][k] * buf2[k][col];

                buf3[row][col] = sum;
                FNorm[0] += (sum*sum);
            });
        }).wait();


        FNorm[0] = std::sqrt(FNorm[0]);


        q.submit([&] (handler &h)
        {
            accessor buf3(vector3_device,h);

            h.parallel_for(range<2>(N,M), [=](auto index){

                int row = index.get_id(0);
                int col = index.get_id(1);

                buf3[row][col] = buf3[row][col]/FNorm[0];
            });
        }).wait();
        
	end = rdtsc();

        elapsed_count[count] = (double)(end-start)/ClkPerSec;
        std::printf("TTC : %.12f\n",elapsed_count[count]);
    }

    // Blocking call to ensure the result is read, after kernel has finished computing.

    host_accessor result(vector3_device,read_only);

    for(int count=0;count<10;count++)
        Average += elapsed_count[count];

    Average = Average/10.0;
    printf("\nTime to compute Matrix Product (Computation without double copy) = %0.12f \n",Average);
    return 0;
}
