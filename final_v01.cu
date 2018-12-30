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

__global__ void computeHist(uint32_t * in, int * hist, int n, int nBins, int bit)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        int bin = (in[idx] >> bit) & (nBins - 1);
        int histIdx = blockIdx.x * nBins + bin;
        atomicAdd(&hist[histIdx], 1);

        //__syncthreads();

        // if (threadIdx.x == 0) {
        //     int baseHistIdx = blockIdx.x * nBins;
        //     int prevBin = hist[baseHistIdx];
        //     int curScan = 0;
        //     hist[baseHistIdx] = curScan;
        //     for (int i = 1; i < nBins; i++) {
        //         curScan += prevBin;
        //         prevBin = hist[baseHistIdx + i];
        //         hist[baseHistIdx + i] = curScan;
        //     }
        // }
    }
}

// Compute hist by device
void computeHistByDevice(uint32_t * in, int * hist, int n, int nBins, int bit, int blkSize) 
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
    computeHist<<<numBlks, blkSize>>>(d_in, d_hist, n, nBins, bit);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(hist, d_hist, sizeof(int) * nBins * numBlks, cudaMemcpyDeviceToHost));
        
    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_hist));
}

__global__ void scanBlks(int * in, int * out, int n, int * blkSums)
{
    // TODO
	// 1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[];
    int i1 = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    int i2 = i1 + blockDim.x;
    if (i1 < n)
        s_data[threadIdx.x] = i1 == 0 ? 0 : in[i1-1];
    if (i2 < n)
        s_data[threadIdx.x + blockDim.x] = in[i2-1];
    __syncthreads();

    // 2. Each block does scan with data on SMEM
    // 2.1. Reduction phase
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
    {
        int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // To avoid warp divergence
        if (s_dataIdx < 2 * blockDim.x)
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        __syncthreads();
    }
    // 2.2. Post-reduction phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride; // Wow
        if (s_dataIdx < 2 * blockDim.x)
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        __syncthreads();
    }

    // 3. Each block writes results from SMEM to GMEM
    if (i1 < n)
        out[i1] = s_data[threadIdx.x];
    if (i2 < n)
        out[i2] = s_data[threadIdx.x + blockDim.x];

    if (blkSums != NULL && threadIdx.x == 0)
        blkSums[blockIdx.x] = s_data[2 * blockDim.x - 1];
}

__global__ void addPrevSum(int * blkSumsScan, int * blkScans, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
	if (i < n)
	{
		blkScans[i] += blkSumsScan[blockIdx.x];
	}
}

// Scan by device
void scanByDevice(int * in, int * out, int n, int blkSize)
{
    // Allocate device memories
    int *d_in, *d_out;
    size_t bytes = n * sizeof(int);
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    int blkDataSize;
    blkDataSize = 2 * blkSize;
    int * d_blkSums;
    int numBlks = (n - 1) / blkDataSize + 1;
    CHECK(cudaMalloc(&d_blkSums, numBlks * sizeof(int)));
    
    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

    // Call kernel to scan within each block's input data
    scanBlks<<<numBlks, blkSize, blkDataSize * sizeof(int)>>>(d_in, d_out, n, d_blkSums);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    
    // Scan "d_blkSums" (by host)
    int * blkSums;
    blkSums = (int *)malloc(numBlks * sizeof(int));
    CHECK(cudaMemcpy(blkSums, d_blkSums, numBlks * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 1; i < numBlks; i++)
        blkSums[i] += blkSums[i-1];
    CHECK(cudaMemcpy(d_blkSums, blkSums, numBlks * sizeof(int), cudaMemcpyHostToDevice));
    free(blkSums);
    
    // Call kernel to add block's previous sum to block's scan result
    printf("numBlks: %d, blkDataSize: %d\n", numBlks, blkDataSize);
    addPrevSum<<<numBlks - 1, blkDataSize>>>(d_blkSums, d_out, n);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
        
    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_blkSums));
}

__global__ void scatter(uint32_t * in, int * histScan, uint32_t * out, int n, int nBins, int bit)
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

void scatterByDevice(uint32_t * in, int * histScan, uint32_t * out, int n, int nBins, int bit, int blkSize)
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
    scatter<<<numBlks, blkSize>>>(d_in, d_histScan, d_out, n, nBins, bit);
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
        computeHistByDevice(src, hist, n, nBins, bit, blkSize);

        int curScan = 0;
        for (int binIdx = 0; binIdx < nBins; binIdx++) {
            for (int blkIdx = 0; blkIdx < numBlks; blkIdx++) {
                int histIdx = blkIdx * nBins + binIdx;
                histScan[histIdx] = curScan;
                curScan += hist[histIdx];
            }
        }

    	// Scan histogram (exclusively)
    	// memset(histScan, 0, nBins * sizeof(int));
    	// for (int bin = 1; bin < nBins; bin++)
    	// 	histScan[bin] = histScan[bin - 1] + hist[bin - 1];

    	// Scatter
    	scatterByDevice(src, histScan, dst, n, nBins, bit, blkSize);

        int * histHost = (int *)malloc(nBins * sizeof(int));
        int * histScanHost = (int *)malloc(nBins * sizeof(int));
        uint32_t * dstHost = (uint32_t *)malloc(n * sizeof(uint32_t));
        // Compute histogram
    	memset(histHost, 0, nBins * sizeof(int));
    	for (int i = 0; i < n; i++)
    	{
    		int bin = (src[i] >> bit) & (nBins - 1);
    		histHost[bin]++;
    	}

    	// Scan histogram (exclusively)
    	memset(histScanHost, 0, nBins * sizeof(int));
    	for (int bin = 1; bin < nBins; bin++)
    		histScanHost[bin] = histScanHost[bin - 1] + histHost[bin - 1];

    	// Scatter
    	for (int i = 0; i < n; i++)
    	{
    		int bin = (src[i] >> bit) & (nBins - 1);
    		dstHost[histScanHost[bin]] = src[i];
    		histScanHost[bin]++;
    	}

        bool check = checkCorrectInt32(dst, dstHost, n);
        if (!check){
            printf("False at bit: %d\n", bit);
            return;
        }

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