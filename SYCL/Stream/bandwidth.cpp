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
    unsigned long long int ClkPerSec;
    double NSecClk;
    unsigned long int host_start,host_end,dev_start,dev_end;
    double elapsed_host1[10],elapsed_dev[10],elapsed_host2[10],HAverage1 = 0.0, HAverage2 = 0.0, DAverage = 0.0;
    queue q(default_selector_v);

    double *source         = static_cast<double*>(malloc(4*1024*1024*sizeof(double)));
    auto *destination      = static_cast<double*>(malloc_device<double>(4*1024*1024,q));
    auto *copy_destination = static_cast<double*>(malloc_device<double>(4*1024*1024,q));

    for(int i=0;i<(4*1024*1024);i++)
        source[i] = 10.0;

    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    Calibrate(&ClkPerSec,NSecClk);

    for(int count=0;count<10;count++)
    {
        host_start = rdtsc();
        auto hostcopy = q.memcpy(destination,source,(sizeof(double)*4*1024*1024));
	hostcopy.wait();
        host_end = rdtsc();
        elapsed_host1[count] = (double)(host_end - host_start)/ClkPerSec;

        dev_start = rdtsc();
        q.parallel_for(range<1>(4*1024*1024), [=](auto index){
            copy_destination[index] = destination[index];
        }).wait();
        dev_end = rdtsc();
        elapsed_dev[count] = (double)(dev_end - dev_start)/ClkPerSec;

        host_start = rdtsc();
        auto host2copy = q.memcpy(source,copy_destination,(sizeof(double)*4*1024*1024));
        host2copy.wait();
        host_end = rdtsc();
        elapsed_host2[count] = (double)(host_end - host_start)/ClkPerSec;
    }

    for(int i = 0;i<10;i++)
    {
        HAverage1 += elapsed_host1[i];
        HAverage2 += elapsed_host2[i];
        DAverage  += elapsed_dev[i];
    }
    HAverage1 = HAverage1/10;
    HAverage2 = HAverage2/10;
    DAverage  = DAverage/10;
    
    std::cout << "Host Transfer Bandwidth   : " << (4)/(HAverage1*1024) << " GB/s\n"
              << "Device Bandwidth          : " << (4)/(DAverage*1024) << " GB/s\n"
              << "Device Transfer Bandwidth : " << (4)/(HAverage2*1024)  << " GB/s" << std::endl;
 
    return 0;
}
