#include <stdio.h>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed;

    GpuTimer()
    {
        elapsed = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
        float timer;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timer, start, stop);
        elapsed += timer;
    }

    float Elapsed()
    {
        return elapsed;
    }
};

bool checkCorrect(int * out, int * correctOut, int n)
{
    for (int i = 0; i < n; i++)
        if (out[i] != correctOut[i]) {
            printf("%d\n", i);
            printf("%d >< %d\n", out[i], correctOut[i]);
            return false;
        }
    return true;
}

bool checkCorrectInt32(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
        if (out[i] != correctOut[i]) {
            printf("%d\n", i);
            printf("%d >< %d\n", out[i], correctOut[i]);
            return false;
        }
    return true;
}

// Sequential Radix Sort
void sortByHost(const uint32_t * in, uint32_t * out, int n)
{
	GpuTimer timer; 
    timer.Start();

    int nBits = 4; // Assume: nBits in {1, 2, 4, 8, 16, 32}
    int nBins = 1 << nBits; // 2^nBits

    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// Compute histogram
    	memset(hist, 0, nBins * sizeof(int));
    	for (int i = 0; i < n; i++)
    	{
    		int bin = (src[i] >> bit) & (nBins - 1);
    		hist[bin]++;
    	}

    	// Scan histogram (exclusively)
    	memset(histScan, 0, nBins * sizeof(int));
    	for (int bin = 1; bin < nBins; bin++)
    		histScan[bin] = histScan[bin - 1] + hist[bin - 1];

    	// Scatter
    	for (int i = 0; i < n; i++)
    	{
    		int bin = (src[i] >> bit) & (nBins - 1);
    		dst[histScan[bin]] = src[i];
    		histScan[bin]++;
    	}

    	// Swap src and dst
    	uint32_t * temp = src;
    	src = dst;
    	dst = temp;
    }

    // Copy result from src to out
    memcpy(out, src, n * sizeof(uint32_t));

	timer.Stop();
    printf("Time of sortByHost: %.3f ms\n\n", timer.Elapsed());
}

__global__ void computeHist(uint32_t * in, int n, int * hist, int nBins, int bit)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        int bin = (in[idx] >> bit) & (nBins - 1);
        int histIdx = blockIdx.x * nBins + bin;
        atomicAdd(&hist[histIdx], 1);
    }
}

// Compute hist by device
void computeHistByDevice(uint32_t * in, int n, int * hist, int nBins, int bit, int blkSize) 
{
    // Allocate device memories
    uint32_t *d_in;
    int *d_hist;
    int numBlks = (n - 1) / blkSize + 1;
    CHECK(cudaMalloc(&d_in, sizeof(uint32_t) * n));
    CHECK(cudaMalloc(&d_hist, sizeof(int) * nBins * numBlks));
    
    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));

    // Call kernel to scan within each block's input data
    computeHist<<<numBlks, blkSize>>>(d_in, n, d_hist, nBins, bit);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(hist, d_hist, sizeof(int) * nBins * numBlks, cudaMemcpyDeviceToHost));
        
    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_hist));
}

__global__ void scatter(uint32_t * in, uint32_t * out, int n, int * histScan, int nBins, int bit)
{
    int idx = blockIdx.x * blockDim.x;
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++){
            if (idx + i < n) {
                int bin = (in[idx + i] >> bit) & (nBins - 1);
                int histIdx = blockIdx.x * nBins + bin;
                out[histScan[histIdx]] = in[idx + i];
                histScan[histIdx]++;
            }
        }
    }
}

void scatterByDevice(uint32_t * in, uint32_t * out, int n, int * histScan, int nBins, int bit, int blkSize)
{
    // Allocate device memories
    uint32_t *d_in, *d_out;
    int *d_histScan;
    int numBlks = (n - 1) / blkSize + 1;
    CHECK(cudaMalloc(&d_in, sizeof(uint32_t) * n));
    CHECK(cudaMalloc(&d_histScan, sizeof(int) * nBins * numBlks));
    CHECK(cudaMalloc(&d_out, sizeof(uint32_t) * n));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_histScan, histScan, sizeof(int) * nBins * numBlks, cudaMemcpyHostToDevice));

    // Call kernel to scan within each block's input data
    scatter<<<numBlks, blkSize>>>(d_in, d_out, n, d_histScan, nBins, bit);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(out, d_out, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost));
        
    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_histScan));
    CHECK(cudaFree(d_out));
}

// Parallel Radix Sort with k bit
void sortByDeviceLv1(const uint32_t * in, uint32_t * out, int n, int blkSize, int nBits)
{
    GpuTimer timer, computeHistTimer, scanTimer, scatterTimer; 
    timer.Start();

    //int nBits = 4; // Assume: nBits in {1, 2, 4, 8, 16, 32}
    int nBins = 1 << nBits; // 2^nBits

    int numBlks = (n - 1) / blkSize + 1;

    int * hist = (int *)malloc(nBins * numBlks * sizeof(int));
    int * histScan = (int *)malloc(nBins * numBlks * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// Compute histogram
        computeHistTimer.Start();
    	memset(hist, 0, nBins * numBlks * sizeof(int));
        computeHistByDevice(src, n, hist, nBins, bit, blkSize);
        computeHistTimer.Stop();

        scanTimer.Start();
        int curScan = 0;
        for (int binIdx = 0; binIdx < nBins; binIdx++) {
            for (int blkIdx = 0; blkIdx < numBlks; blkIdx++) {
                int histIdx = blkIdx * nBins + binIdx;
                histScan[histIdx] = curScan;
                curScan += hist[histIdx];
            }
        };
        scanTimer.Stop();

    	// Scatter
        scatterTimer.Start();
    	scatterByDevice(src, dst, n, histScan, nBins, bit, blkSize);
        scatterTimer.Stop();

    	// Swap src and dst
    	uint32_t * temp = src;
    	src = dst;
    	dst = temp;
    }

    // Copy result from src to out
    memcpy(out, src, n * sizeof(uint32_t));
    
    timer.Stop();
    printf("Time of sortByDevice Level 1: %.3f ms\n", timer.Elapsed());
    printf("Time of compute hist: %.3f ms\n", computeHistTimer.Elapsed());
    printf("Time of scan hist: %.3f ms\n", scanTimer.Elapsed());
    printf("Time of scatter hist: %.3f ms\n", scatterTimer.Elapsed());
    printf("\n");
}

__global__ void computeHistLocalSort(uint32_t * in, int n, int * hist, int nBits, int bit) 
{
    int nBins = 1 << nBits; // 2^nBits
    int baseIdx = blockDim.x * blockIdx.x;
    int idx = baseIdx + threadIdx.x;

    extern __shared__ uint32_t s_data[];
    int baseCopyIdx = blockDim.x;
    int baseHistIdx = blockDim.x * 2;

    int blockSize;
    if (blockIdx.x == gridDim.x - 1)
        blockSize = n - baseIdx;
    else 
        blockSize = blockDim.x;

    
    if (idx < n) {
        // 1. Input current data to block mem
        s_data[threadIdx.x] = in[idx];
        __syncthreads();
    
        // 2. Local sort
        for (int b = 0; b < nBits; b++) {
            // 2.1. Scan numZerosBefore
            s_data[baseCopyIdx + threadIdx.x] = (s_data[threadIdx.x] >> (bit+b)) & 1;
            __syncthreads();
            
            s_data[baseHistIdx + threadIdx.x] = threadIdx.x == 0 ? 0 : s_data[baseCopyIdx + threadIdx.x - 1];
            __syncthreads();
            for (int stride = 1; stride < blockDim.x; stride *= 2){
                int temp = s_data[baseHistIdx + threadIdx.x];
                if (threadIdx.x >= stride) {
                    temp = s_data[baseHistIdx + threadIdx.x] + s_data[baseHistIdx + threadIdx.x - stride];
                }
                __syncthreads();
                s_data[baseHistIdx + threadIdx.x] = temp ;
                __syncthreads();
            }

            // 2.2. Get numZeros of current block
            int numZeros = blockSize - s_data[baseHistIdx + blockSize-1] - s_data[baseCopyIdx + blockSize-1];

            // 2.3. Calculate rank
            int rank;
            if (s_data[baseCopyIdx + threadIdx.x] == 0)
                rank = threadIdx.x - s_data[baseHistIdx + threadIdx.x];
            if (s_data[baseCopyIdx + threadIdx.x] == 1)
                rank = numZeros + s_data[baseHistIdx + threadIdx.x];
            __syncthreads();

            s_data[baseHistIdx + threadIdx.x] = rank;
            
            // 2.4. Local scatter
            s_data[baseCopyIdx + s_data[baseHistIdx + threadIdx.x]] = s_data[threadIdx.x];
            __syncthreads();
            s_data[threadIdx.x] = s_data[baseCopyIdx + threadIdx.x];
            __syncthreads();
        }

        // 3. Copy value to host
        in[idx] = s_data[threadIdx.x];
        
        // 4. Compute hist
        int bin = (s_data[threadIdx.x] >> bit) & (nBins - 1);
        int histIdx = blockIdx.x * nBins + bin;
        atomicAdd(&hist[histIdx], 1);
    }
}

// Compute hist by device
void computeHistLocalSortByDevice(uint32_t * in, int n, int * hist, int nBits, int bit, int blkSize) 
{
    int nBins = 1 << nBits; // 2^nBits
    // Allocate device memories
    uint32_t *d_in;
    int *d_hist;
    int numBlks = (n - 1) / blkSize + 1;
    int sharedMem = blkSize * (sizeof(uint32_t)*3);
    CHECK(cudaMalloc(&d_in, sizeof(uint32_t) * n));
    CHECK(cudaMalloc(&d_hist, sizeof(int) * nBins * numBlks));
    
    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));

    // Call kernel to scan within each block's input data
    computeHistLocalSort<<<numBlks, blkSize, sharedMem>>>(d_in, n, d_hist, nBits, bit);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(hist, d_hist, sizeof(int) * nBins * numBlks, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(in, d_in, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost));
        
    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_hist));
}

__global__ void scatterLocalSort(uint32_t * in, uint32_t * out, int n, int * hist, int * histScan, int nBins, int bit)
{
    int baseIdx = blockIdx.x * blockDim.x;
    int idx = baseIdx + threadIdx.x;

    extern __shared__ int localHistScan[];
    if (threadIdx.x == 0) {
        localHistScan[0] = 0;
        for (int binIdx = 1; binIdx < nBins; binIdx++) {
            int histIdx = blockIdx.x * nBins + binIdx - 1;
            localHistScan[binIdx] = localHistScan[binIdx - 1] + hist[histIdx]; 
        }
    }
    __syncthreads();

    if (idx < n) {
        int bin = (in[idx] >> bit) & (nBins - 1);
        int histIdx = blockIdx.x * nBins + bin;
        int rank = histScan[histIdx] + threadIdx.x - localHistScan[bin];
        out[rank] = in[idx];
    }
}

void scatterLocalSortByDevice(uint32_t * in, uint32_t * out, int n, int * hist, int * histScan, int nBins, int bit, int blkSize)
{
    // Allocate device memories
    uint32_t *d_in, *d_out;
    int *d_histScan, *d_hist;
    int numBlks = (n - 1) / blkSize + 1;
    CHECK(cudaMalloc(&d_in, sizeof(uint32_t) * n));
    CHECK(cudaMalloc(&d_out, sizeof(uint32_t) * n));
    CHECK(cudaMalloc(&d_histScan, sizeof(int) * nBins * numBlks));
    CHECK(cudaMalloc(&d_hist, sizeof(int) * nBins * numBlks));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_histScan, histScan, sizeof(int) * nBins * numBlks, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hist, hist, sizeof(int) * nBins * numBlks, cudaMemcpyHostToDevice));

    // Call kernel to scan within each block's input data
    scatterLocalSort<<<numBlks, blkSize, nBins * sizeof(int)>>>(d_in, d_out, n, d_hist, d_histScan, nBins, bit);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(out, d_out, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost));
        
    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
}

// Parallel Radix Sort with k bit
void sortByDeviceLv2(const uint32_t * in, uint32_t * out, int n, int blkSize, int nBits)
{
	GpuTimer timer, computeHistTimer, scanTimer, scatterTimer; 
    timer.Start();

    //int nBits = 4; // Assume: nBits in {1, 2, 4, 8, 16, 32}
    int nBins = 1 << nBits; // 2^nBits

    int numBlks = (n - 1) / blkSize + 1;

    int * hist = (int *)malloc(nBins * numBlks * sizeof(int));
    int * histScan = (int *)malloc(nBins * numBlks * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// Compute histogram
        computeHistTimer.Start();
    	memset(hist, 0, nBins * numBlks * sizeof(int));
        computeHistLocalSortByDevice(src, n, hist, nBits, bit, blkSize);
        computeHistTimer.Stop();

        scanTimer.Start();
        int curScan = 0;
        for (int binIdx = 0; binIdx < nBins; binIdx++) {
            for (int blkIdx = 0; blkIdx < numBlks; blkIdx++) {
                int histIdx = blkIdx * nBins + binIdx;
                histScan[histIdx] = curScan;
                curScan += hist[histIdx];
            }
        }
        scanTimer.Stop();

        // Scatter
        scatterTimer.Start();
    	scatterLocalSortByDevice(src, dst, n, hist, histScan, nBins, bit, blkSize);
        scatterTimer.Stop();

    	// Swap src and dst
    	uint32_t * temp = src;
    	src = dst;
    	dst = temp;
    }

    // Copy result from src to out
    memcpy(out, src, n * sizeof(uint32_t));
    
    timer.Stop();
    printf("Time of sortByDevice Level 2: %.3f ms\n", timer.Elapsed());
    printf("Time of compute hist: %.3f ms\n", computeHistTimer.Elapsed());
    printf("Time of scan hist: %.3f ms\n", scanTimer.Elapsed());
    printf("Time of scatter hist: %.3f ms\n", scatterTimer.Elapsed());
    printf("\n");
}

// Parallel Radix Sort with k = 1 bit
void sortByDevice(const uint32_t * in, uint32_t * out, int n, int blockSize)
{
	GpuTimer timer; 
    timer.Start();
	
	// TODO
    thrust::device_vector<uint32_t> dv_out(in, in + n);
    thrust::sort(dv_out.begin(), dv_out.end());
    thrust::copy(dv_out.begin(), dv_out.end(), out);
    
	timer.Stop();
    printf("Time of sortByDevice by thrust: %.3f ms\n\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    printf("Input size: %d\n\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    int nBits = 4;
    if (argc >= 2)
        blockSize = atoi(argv[1]);

    if (argc == 3)
        nBits = atoi(argv[2]);

    // SORT BY HOST
    sortByHost(in, correctOut, n);
    
    // SORT BY DEVICE LEVEL 1
    sortByDeviceLv1(in, out, n, blockSize, nBits);
    if (checkCorrectInt32(out, correctOut, n) == false)
        printf("sortByDevice is INCORRECT!\n\n");

    // SORT BY DEVICE LEVEL 2
    sortByDeviceLv2(in, out, n, blockSize, nBits);
    if (checkCorrectInt32(out, correctOut, n) == false)
        printf("sortByDevice is INCORRECT!\n\n");
        
    // SORT BY DEVICE by THRUST
    sortByDevice(in, out, n, blockSize);
    if (checkCorrectInt32(out, correctOut, n) == false)
    	printf("sortByDevice is INCORRECT!\n\n");

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}