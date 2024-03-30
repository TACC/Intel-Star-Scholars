//#include <SYCL/sycl.hpp>
#include <sycl/sycl.hpp>
#include <chrono>

//using namespace hipsycl::sycl;
using namespace sycl;

int main()
{
    const int N=2048,M=256;

    // Using time point and system_clock
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds[10],Average;

    queue q(default_selector_v);

    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    // Initialize Vectors and Print Values
    int *vector1 = static_cast<int*>(malloc(N*M*sizeof(int)));
    int *vector2 = static_cast<int*>(malloc(N*M*sizeof(int)));

    for(int i=0;i<N*M;i++)
    {
        vector1[i] = 10;
        vector2[i] = 20;
    }

    /*std::cout << "\nInput Vector1: \n";
    for(int i=0;i<(N*M);i+=M)
    {
        for(int j=i;j<(i+M);j++)
            std::cout << vector1[j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nInput Vector2: \n";
    for(int i=0;i<(N*M);i+=M)
    {
        for(int j=i;j<(i+M);j++)
            std::cout << vector2[j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
    */

    // Initialize Device USM and copy host data over to device for processing

    auto *vector1_device = static_cast<int*>(malloc_device<int>(N*M,q));
    auto *vector2_device = static_cast<int*>(malloc_device<int>(N*M,q));

    for(int count=0;count<10;count++)
    {
        start = std::chrono::system_clock::now();

        auto e1 = q.memcpy(vector1_device,vector1,(sizeof(int)*N*M));
        auto e2 = q.memcpy(vector2_device,vector2,(sizeof(int)*N*M));

    // Kernel to add the two Two-Dim Vectors
        auto e3 = q.parallel_for(range<1>(N*M), {e1,e2}, [=](auto index){
            vector1_device[index] += vector2_device[index];
        });

        q.memcpy(vector1,vector1_device,sizeof(int)*N*M,e3).wait();

        end = std::chrono::system_clock::now();

        elapsed_seconds[count] = end - start;
    }

    // Print updated Vector1 after Sum

    /*std::cout << "\nUpdated Input Vector1: \n";
    for(int i=0;i<N*M;i+=M)
    {
        for(int j=0;j<M;j++)
            std::cout << vector1[j] << " ";
        std::cout << "\n";
    }*/

    for(int count=0;count<10;count++)
        Average += elapsed_seconds[count];

    Average = Average/10.0;
    std::cout << "\nTime to compute Matrix Sum (Copy + Computation + Copy) = " << Average.count() << "\n";

    free(vector1_device,q);
    free(vector2_device,q);
    free(vector1);
    free(vector2);

    return 0;
}
