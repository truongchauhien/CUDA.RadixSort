#include <stdio.h>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#define DEBUG 0

typedef enum { SORT_BY_HOST, SORT_BY_THRUST, SORT_BY_DEVICE } Implementation;

#define CHECK(call) {                                                          \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void sortByHost(const uint32_t *input, int n, uint32_t *output, int nBits) {
    int nBins = 1 << nBits;

    size_t inputMemSize = n * sizeof(uint32_t);
    uint32_t *in = (uint32_t *)malloc(inputMemSize);
    uint32_t *originalIn = in;
    memcpy(in, input, inputMemSize);
    uint32_t *out = output;

    int *hist = (int *)malloc(nBins * sizeof(int));
    int *histScan = (int *)malloc(nBins * sizeof(int));

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
    	// TODO: Compute histogram
        memset(hist, 0, nBins * sizeof(int));
        for (int index = 0; index < n; index++) {
            int bin = (in[index] >> bit) & (nBins - 1);
            hist[bin]++;
        }

    	// TODO: Scan histogram (exclusively)
        histScan[0] = 0;
        for (int bin = 1; bin < nBins; bin++) {
            histScan[bin] = histScan[bin - 1] + hist[bin - 1];
        }

    	// TODO: Scatter elements to correct locations
        for (int index = 0; index < n; index++) {
            int bin = (in[index] >> bit) & (nBins - 1);
            out[histScan[bin]] = in[index];
            histScan[bin]++;
        }
    	
    	// Swap in and out.
        uint32_t *temp = in;
        in = out;
        out = temp;
    }
    memcpy(output, in, n * sizeof(uint32_t)); 

    free(hist);
    free(histScan);
    free(originalIn);    
}

void sortByThrust(const uint32_t *in, int n, uint32_t *out) {
    thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}

__global__ void calculateHistogram(const uint32_t *in, int n, int *hist, int nBins, int bit) {
    // SMEM size: nBins elements.
    extern __shared__ int s_hist[];

    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
        s_hist[bin] = 0;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        int bin = (in[index] >> bit) & (nBins - 1);
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
        atomicAdd(&hist[bin], s_hist[bin]);
    }
}

__global__ void scanPerBlock(const uint32_t *input, int n, int *output, int bit, int *blockSums) {
    // SMEM Size: blockDim.x elements.
    extern __shared__ int s_data[];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Load data handled by this block into SMEM.
    if (index < n) {
        s_data[threadIdx.x] = (input[index] >> bit) & 1;
    } else {
        s_data[threadIdx.x] = 0;
    }
    __syncthreads();

    // 2. Do scan with data on SMEM. Implementation of Work-Efficient algorithm.
    // >>>> Up-Sweep phase.
    int offset = 1; // Distance between 2 adjacent nodes.
    for (int nNodes = blockDim.x / 2; nNodes > 0; nNodes /= 2) {
        offset *= 2;
        if (threadIdx.x < nNodes) {
            s_data[threadIdx.x * offset + offset - 1]
                += s_data[threadIdx.x * offset + (offset / 2) - 1];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockSums != NULL) {
        blockSums[blockIdx.x] = s_data[blockDim.x - 1];
    }
    if (threadIdx.x == 0) {
        s_data[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // >>>> Down-Sweep phase.
    int temp;
    for (int nNodes = 1; nNodes < blockDim.x; nNodes *= 2) {
        if (threadIdx.x < nNodes) {
            temp = s_data[threadIdx.x * offset + (offset / 2) - 1];
            s_data[threadIdx.x * offset + (offset / 2) - 1] 
                = s_data[threadIdx.x * offset + offset - 1];
            s_data[threadIdx.x * offset + offset - 1] += temp;
        }
        offset /= 2;
        __syncthreads();
    }

    // 3. Copy back results from SMEM to GMEM.
    if (index < n) {
        output[index] = s_data[threadIdx.x];
    }
}

__global__ void addScannedBlockSumsToScannedBlocks(int *blockSums, int *blockScans, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    index += blockDim.x; // Shift to right 1 block.
    if (index < n) {
        blockScans[index] += blockSums[blockIdx.x];
    }
}

__global__ void scatter(const uint32_t *g_input, int n, int *g_inputScan, int bitOrder, uint32_t *g_output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        int nZeros = n - g_inputScan[n - 1] - ((g_input[n - 1] >> bitOrder) & 1);
        int bit = (g_input[index] >> bitOrder) & 1;

        int rank;
        if (bit == 0) {
            rank = index - g_inputScan[index];
        } else {
            rank = nZeros + g_inputScan[index];
        }
        
        g_output[rank] = g_input[index];
    }
}

void sortByDevice(const uint32_t *input, int n, uint32_t *output, int blockSize) {
    uint32_t *d_input = NULL;
    uint32_t *d_output = NULL;
    size_t inputMemSize = n * sizeof(uint32_t);    
    CHECK(cudaMalloc(&d_input, inputMemSize));
    CHECK(cudaMalloc(&d_output, inputMemSize));
    CHECK(cudaMemcpy(d_input, input, inputMemSize, cudaMemcpyHostToDevice));

    int *d_inputScan = NULL;
    int *blockSums = NULL, *d_blockSums = NULL;
    dim3 scanBlockDim(blockSize);
    dim3 scanGridDim((n - 1) / scanBlockDim.x + 1);
    dim3 addScannedBlockSumsToScannedBlocksBlockDim(scanBlockDim.x);
    dim3 addScannedBlockSumsToScannedBlocksGridDim(scanGridDim.x - 1);
    size_t inputScanMemSize = n * sizeof(int);
    size_t inputScanSMEMSize = scanBlockDim.x * sizeof(int);
    size_t blockSumsMemSize = 0;
    CHECK(cudaMalloc(&d_inputScan, inputScanMemSize));
    if (scanGridDim.x > 1) {
        blockSumsMemSize = scanGridDim.x * sizeof(int);
        blockSums = (int *)malloc(blockSumsMemSize);
        CHECK(cudaMalloc(&d_blockSums, blockSumsMemSize));
    }

    dim3 scatterBlockDim(blockSize);
    dim3 scatterGridDim((n - 1) / scatterBlockDim.x + 1);

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += 1) {
    	// Exclusive Scan on digits (with 1-bit length) at position "bit" of elements in "in" array.
        scanPerBlock<<<scanGridDim, scanBlockDim, inputScanSMEMSize>>>(d_input, n, d_inputScan, bit, d_blockSums);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        if (scanGridDim.x > 1) {
            CHECK(cudaMemcpy(blockSums, d_blockSums, blockSumsMemSize, cudaMemcpyDeviceToHost));
            for (int index = 1; index < scanGridDim.x; ++index) {
                blockSums[index] += blockSums[index - 1];
            }
            CHECK(cudaMemcpy(d_blockSums, blockSums, blockSumsMemSize, cudaMemcpyHostToDevice));

            addScannedBlockSumsToScannedBlocks<<<
                addScannedBlockSumsToScannedBlocksGridDim,
                addScannedBlockSumsToScannedBlocksBlockDim
            >>>(d_blockSums, d_inputScan, n);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
        }

        // Calculate rank and scatter.
        scatter<<<scatterGridDim, scatterBlockDim>>>(d_input, n, d_inputScan, bit, d_output);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    	
        // Preventing copying memory by recycle memory allocations.
        uint32_t *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    cudaMemcpy(output, d_input, inputMemSize, cudaMemcpyDeviceToHost);

    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_inputScan));
    CHECK(cudaFree(d_blockSums));
    free(blockSums);
}

void sort(const uint32_t * in, int n,
          uint32_t * out,
          Implementation implementation = SORT_BY_HOST,
          int nBits = 4,
          int blockSize = 1) {
    GpuTimer timer; 
    timer.Start();

    if (implementation == SORT_BY_HOST) {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out, nBits);
    } else if (implementation == SORT_BY_THRUST) {
    	printf("\nRadix Sort by Thrust library\n");
        sortByThrust(in, n, out);
    } else {
        printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo() {
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
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != correctOut[i]) {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%i ", a[i]);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n;
    if (DEBUG) {
        n = 513;
    } else {
        n = (1 << 24) + 1;
    }
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t *in = (uint32_t *)malloc(bytes);
    uint32_t *out = (uint32_t *)malloc(bytes);
    uint32_t *correctOut = (uint32_t *)malloc(bytes);

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++) {
        if (DEBUG) {
            in[i] = rand() & 0xFF;
        } else {
            in[i] = rand();
        }
    }
    if (DEBUG) {
        printArray(in, n);
    }

    // DETERMINE BLOCK SIZE
    int blockSize = 512;
    if (argc == 2) {
        blockSize = atoi(argv[1]);
    }

    int nBits = 4;
    if (argc == 3) {
        nBits = atoi(argv[2]);
    }

    // Sorting by Host
    sort(in, n, correctOut, SORT_BY_HOST, nBits);
    if (DEBUG) {
        printArray(correctOut, n);
    }

    // Sorting by Thrust Library
    sort(in, n, out, SORT_BY_THRUST);
    if (DEBUG) {
        printArray(out, n);
    }
    checkCorrectness(out, correctOut, n);
    memset(out, 0u, n * sizeof(uint32_t)); // Reset ouput.

    // Sorting by Device
    sort(in, n, out, SORT_BY_DEVICE, nBits, blockSize);
    if (DEBUG) {
        printArray(out, n);
    }
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
