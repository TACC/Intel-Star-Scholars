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
    std::vector<int> vector1(N*M,10);
    std::vector<int> vector2(N*M,20);

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

    buffer vector1_device(vector1.data(),range<1>(vector1.size()));
    buffer vector2_device(vector2.data(),range<1>(vector2.size()));
    // Shared Unified Memory created, without the need for copy

    for(int count=0;count<10;count++)
    {
        start = std::chrono::system_clock::now();

        // Kernel to add the two Two-Dim Vectors
        q.submit([&] (handler &h) 
        {
            accessor device1(vector1_device,h);
            accessor device2(vector2_device,h);

            h.parallel_for(range<1>(N*M), [=](auto index){
            device1[index] += device2[index];
            });
        });

        // Blocking call to ensure the result is read, after kernel has finished computing.

        host_accessor result(vector1_device,read_only);

        end = std::chrono::system_clock::now();

        elapsed_seconds[count] = end - start;
    }
    // Print updated Vector1 after Sum


    /*std::cout << "\nUpdated Input Vector1: \n";
    for(int i=0;i<N*M;i+=M)
    {
        for(int j=0;j<M;j++)
            std::cout << result[j] << " ";
        std::cout << "\n";
    }*/


    for(int count=0;count<10;count++)
        Average += elapsed_seconds[count];

    Average = Average/10.0;
    std::cout << "\nTime to compute Matrix Sum (Computation without double copy) = " << Average.count() << "\n";

    return 0;
}
