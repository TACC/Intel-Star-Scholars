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

// Kernel to compute sqrt of FNorm **once**
__global__ void compute_sqrt(float *FNorm) {
    *FNorm = sqrtf(*FNorm);  // Perform single square root
}

// Kernel to normalize the matrix
__global__ void normalize_kernel(float *Mat_A, const float *Mat_Stencil, float *FNorm, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (row < N - 1 && col < M - 1) {
        Mat_A[row * M + col] = Mat_Stencil[(row - 1) * (M - 2) + (col - 1)] / *FNorm;
    }
}

int main(int argc, char *argv[]) {

    int loops = atoi(argv[1]);
    int N = atoi(argv[2]);
    int M = atoi(argv[3]);

    unsigned long long int ClkPerSec;
    double NSecClk;

    std::vector<float> Mat_A(N * M, 1.0f);
    std::vector<float> Mat_Stencil((N - 2) * (M - 2), 0.0f);
    float FNorm = 0.0f;
    double elapsed_time[10], Average = 0.0;

    Calibrate(&ClkPerSec, NSecClk);

    // Allocate device memory
    float *d_Mat_A, *d_Mat_Stencil, *d_FNorm;
    cudaMalloc(&d_Mat_A, N * M * sizeof(float));
    cudaMalloc(&d_Mat_Stencil, (N - 2) * (M - 2) * sizeof(float));
    cudaMalloc(&d_FNorm, sizeof(float));


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int count = 0; count < loops; count++) {

        if(FNorm == 0.0f)
          FNorm = 1.0f;
        else
          FNorm = 1.0f;
        unsigned long long start = rdtsc();
        cudaMemcpy(d_FNorm, &FNorm, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Mat_A, Mat_A.data(), N * M * sizeof(float), cudaMemcpyHostToDevice);


        // Launch the stencil kernel
        stencil_kernel<<<numBlocks, threadsPerBlock>>>(d_Mat_A, d_Mat_Stencil, N, M, d_FNorm);
        cudaDeviceSynchronize();

        // Perform single sqrt calculation
        compute_sqrt<<<1, 1>>>(d_FNorm);  // Single thread kernel
        cudaDeviceSynchronize();

        // Normalize the matrix
        normalize_kernel<<<numBlocks, threadsPerBlock>>>(d_Mat_A, d_Mat_Stencil, d_FNorm, N, M);
        cudaDeviceSynchronize();

        cudaMemcpy(Mat_A.data(), d_Mat_A, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&FNorm, d_FNorm, sizeof(float), cudaMemcpyDeviceToHost);
        unsigned long long end = rdtsc();

        /*for(int i=0;i<N*M;i+=M)
        {
            for(int j=i;j<(i+M);j++)
                printf("%f ",Mat_A[j]);
            printf("\n");
        }*/

        elapsed_time[count] = (double)(end - start) / ClkPerSec;
        printf("TTC : %.12f\n", elapsed_time[count]);
    }


    /*for(int i=0;i<N*M;i+=M)
    {
        for(int j=i;j<(i+M);j++)
            printf("%f ",Mat_A[j]);
        printf("\n");
    }*/ 

    // Print average time
    for (int count = 1; count < loops; count++) {
        Average += elapsed_time[count];
    }
    Average /= (loops - 1);
    printf("Average Computation Time: %.12f\n", Average);

    cudaFree(d_Mat_A);
    cudaFree(d_Mat_Stencil);
    cudaFree(d_FNorm);

    return 0;
}
