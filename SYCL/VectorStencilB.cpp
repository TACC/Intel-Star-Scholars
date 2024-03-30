#include<sycl/sycl.hpp>

using namespace sycl;

int main()
{
    const int N=6,M=6,K=256;
    int index,approx=100;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds[10],Average;

    queue q(default_selector_v);

    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    double *H_a = static_cast<double*>(malloc(N*M*sizeof(double)));
    double *Up_H_a = static_cast<double*>(malloc(N*M*sizeof(double)));
    //double *H_Eigen = static_cast<double*>(malloc_shared(M*1*sizeof(double),q));
    //double *Smallest_Eigen = static_cast<double*>(malloc_shared(sizeof(double),q));
    //Smallest_Eigen[0] = 1.0;

    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            H_a[((i*N) + j)] = (i+j);
            Up_H_a[((i*N) + j)] = (i+j);
        }

    }

    std::cout << "Original Square Domain : \n";
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
            std::cout << H_a[(i*N) + j] << "\t";

        std::cout << "\n";
    }

    //for(int i=0;i<N;i++)
    //    H_Eigen[i] = 1.0;

    auto *D_a = static_cast<double*>(malloc_device<double>(N*M,q));
    //auto *D_Stencil = static_cast<double*>(malloc_device<double>((N-2)*(M-2),q));
    //auto *D_Eigen   = static_cast<double*>(malloc_device<double>(M*1,q));

    start = std::chrono::system_clock::now();

    auto e1 = q.memcpy(D_a,H_a,(sizeof(double)*N*M));

    auto e3 = q.parallel_for(nd_range<1>(N,M), [=](nd_item<1> item)[[intel::reqd_sub_group_size(8)]]{

        auto sg = item.get_sub_group();

      //# query sub_group and print sub_group info once per sub_group
      if (sg.get_local_id()[0] == 0) {
          std::cout << "sub_group id: " << sg.get_group_id()[0] << " of "
              << sg.get_group_range()[0] << ", size=" << sg.get_local_range()[0]
              << "\n";
      }
    });
    e3.wait();
}
