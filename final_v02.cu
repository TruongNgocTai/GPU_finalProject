#include <stdio.h>
#include <stdint.h>

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

    GpuTimer()
    {
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
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
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
            s_data[baseCopyIdx + threadIdx.x] = 0;
            __syncthreads();
            
            // 2.4. Local scatter
            s_data[baseCopyIdx + s_data[baseHistIdx + threadIdx.x]] = s_data[threadIdx.x];
            __syncthreads();
            s_data[threadIdx.x] = s_data[baseCopyIdx + threadIdx.x];
            __syncthreads();
            
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                for (int i = 1; i < blockSize; i++){
                    int bin = (s_data[i] >> (bit+b)) & 1;
                    int lastbin = (s_data[i-1] >> (bit+b)) & 1;
    
                    if (bin < lastbin) {
                        if (blockIdx.x == 0) {
                            printf("\n-----\nDM.%d - %d - %d < %d\n", b, i, bin, lastbin);
                        }
                        break;
                    }
                }
            }
            __syncthreads();
            
        }

        // 3. Copy value to host
        in[idx] = s_data[threadIdx.x];
        __syncthreads();
        
        // 4. Compute hist
        int bin = (s_data[threadIdx.x] >> bit) & (nBins - 1);
        int histIdx = blockIdx.x * nBins + bin;
        atomicAdd(&hist[histIdx], 1);
    }
}

// Compute hist by device
void computeHistByDevice(uint32_t * in, int n, int * hist, int nBits, int bit, int blkSize) 
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

__global__ void scatter(uint32_t * in, uint32_t * out, int n,int * hist, int * histScan, int nBins, int bit)
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

void scatterByDevice(uint32_t * in, uint32_t * out, int n, int * hist, int * histScan, int nBins, int bit, int blkSize)
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
    scatter<<<numBlks, blkSize, nBins * sizeof(int)>>>(d_in, d_out, n, d_hist, d_histScan, nBins, bit);
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
void sortByDevice(const uint32_t * in, uint32_t * out, int n, int blkSize)
{
	GpuTimer timer; 
    timer.Start();

    int nBits = 4; // Assume: nBits in {1, 2, 4, 8, 16, 32}
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
    	memset(hist, 0, nBins * numBlks * sizeof(int));
        computeHistByDevice(src, n, hist, nBits, bit, blkSize);

        int curScan = 0;
        for (int binIdx = 0; binIdx < nBins; binIdx++) {
            for (int blkIdx = 0; blkIdx < numBlks; blkIdx++) {
                int histIdx = blkIdx * nBins + binIdx;
                histScan[histIdx] = curScan;
                curScan += hist[histIdx];
            }
        }

    	// Scatter
    	scatterByDevice(src, dst, n, hist, histScan, nBins, bit, blkSize);

    	// Swap src and dst
    	uint32_t * temp = src;
    	src = dst;
    	dst = temp;
    }

    // Copy result from src to out
    memcpy(out, src, n * sizeof(uint32_t));
    
    timer.Stop();
    printf("Time of sortByDevice: %.3f ms\n\n", timer.Elapsed());
}

bool checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
        if (out[i] != correctOut[i]) {
            printf("%d\n", i);
            return false;
        }
    return true;
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
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sortByHost(in, correctOut, n);
    
    // SORT BY DEVICE
    sortByDevice(in, out, n, blockSize);
    if (checkCorrectness(out, correctOut, n) == false)
    	printf("sortByDevice is INCORRECT!\n\n");;

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}