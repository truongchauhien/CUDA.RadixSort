#include "common/common.h"
#include <stdio.h>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#define DEVICE_ID 0
#define DEBUG 0

typedef enum { SORT_BY_HOST, SORT_BY_THRUST, SORT_BY_DEVICE } Implementation;

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

__global__ void scanPerBlock(const int *input, int n, int *output, int *blockSums) {
    // SMEM Size: blockDim.x elements.
    extern __shared__ int s_data[];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Load data handled by this block into SMEM.
    if (index < n) {
        s_data[threadIdx.x] = input[index];
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

void sortByDevice(const uint32_t *input, int n, uint32_t *output, int nBits, int blockSize) {
    int nBins = 1 << nBits;
    
    size_t inputMemSize = n * sizeof(uint32_t);
    uint32_t *in = (uint32_t *)malloc(inputMemSize);
    uint32_t *originalIn = in;
    uint32_t *out = output;
    memcpy(in, input, inputMemSize);

    uint32_t *d_in = NULL;
    CHECK(cudaMalloc(&d_in, inputMemSize));

    int *d_hist = NULL;
    dim3 histBlock(blockSize);
    dim3 histGrid((n + histBlock.x - 1) / histBlock.x);
    size_t histMemSize = nBins * sizeof(int);
    CHECK(cudaMalloc(&d_hist, histMemSize));

    int *histScan = NULL, *d_histScan = NULL;
    int *blockSums = NULL, *d_blockSums = NULL;
    dim3 scanBlockDim(blockSize);
    dim3 scanGridDim((nBins - 1) / scanBlockDim.x + 1);
    dim3 addScannedBlockSumsToScannedBlocksBlockDim(scanBlockDim.x);
    dim3 addScannedBlockSumsToScannedBlocksGridDim(scanGridDim.x - 1);
    size_t scanSMEMSize = scanBlockDim.x * sizeof(int);
    size_t blockSumsMemSize = 0;
    histScan = (int *)malloc(histMemSize);
    CHECK(cudaMalloc(&d_histScan, histMemSize));
    if (scanGridDim.x > 1) {
        blockSumsMemSize = scanGridDim.x * sizeof(int);
        blockSums = (int *)malloc(blockSumsMemSize);
        CHECK(cudaMalloc(&d_blockSums, blockSumsMemSize));
    }

    // Applying Counting Sort for digits in every position.
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
    	// TODO: Compute histogram
        CHECK(cudaMemcpy(d_in, in, inputMemSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_hist, 0, histMemSize));
        calculateHistogram<<<histGrid, histBlock, histMemSize>>>(d_in, n, d_hist, nBins, bit);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

    	// TODO: Scan histogram (exclusively)
        scanPerBlock<<<scanGridDim, scanBlockDim, scanSMEMSize>>>(d_hist, nBins, d_histScan, d_blockSums);
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
            >>>(d_blockSums, d_histScan, nBins);
        }
        CHECK(cudaMemcpy(histScan, d_histScan, histMemSize, cudaMemcpyDeviceToHost));

    	// TODO: Scatter elements to correct locations
        for (int index = 0; index < n; ++index) {
            int bin = (in[index] >> bit) & (nBins - 1);
            out[histScan[bin]] = in[index];
            histScan[bin]++;
        }
    	
        // Preventing copying memory by recycle memory allocations.
        uint32_t *temp = in;
        in = out;
        out = temp;
    }
    // The memory allocation pointed by variable "output" (not "out"),
    // maybe contains the final results in case of the number of loops is odd (only 1 case).
    // The variable "in" always points to memory allocation that contains the final results,
    // because the swapping at the end of the loop.
    // The memory copying below is to be sure that memory allocation pointed by "output" contains the final results.
    memcpy(output, in, inputMemSize);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_hist));
    free(histScan);
    free(originalIn);
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
        sortByDevice(in, n, out, nBits, blockSize);
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

int main(int argc, char **argv) {
    CHECK(cudaSetDevice(DEVICE_ID));
    CHECK(cudaDeviceReset());
    cudaFree(0);
    
    printDeviceInfo();

    // Initialization of input.
    int n;
    #if DEBUG
    n = 513;
    #else
    n = (1 << 24) + 1;
    #endif
    printf("\nInput size: %d\n", n);

    size_t inputMemSize = n * sizeof(uint32_t);
    uint32_t *input = (uint32_t *)malloc(inputMemSize);
    uint32_t *output = (uint32_t *)malloc(inputMemSize);
    uint32_t *correctOutput = (uint32_t *)malloc(inputMemSize);

    for (int i = 0; i < n; i++) {
        #if DEBUG
        input[i] = rand() & 0xFF;
        #else
        input[i] = rand();
        #endif
    }
    #if DEBUG
    printArray(input, n);
    #endif

    // Block size.
    int blockSize = 512;
    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }
    printf("Block size: %d\n", blockSize);

    // Digit width.
    int numBits;
    #if DEBUG
    numBits = 4;
    #else
    numBits = 8;
    #endif
    if (argc > 2) {
        numBits = atoi(argv[2]);
    }
    printf("Digit width: %d-bit\n", numBits);

    // Sorting by Host
    sort(input, n, correctOutput, SORT_BY_HOST, numBits);
    #if DEBUG
    printArray(correctOutput, n);
    #endif

    // Sorting by Thrust Library.
    memset(output, 0u, inputMemSize);
    sort(input, n, output, SORT_BY_THRUST);
    #if DEBUG
    printArray(output, n);
    #endif
    checkCorrectness(output, correctOutput, n);

    // Sorting by Device.
    memset(output, 0u, inputMemSize);
    sort(input, n, output, SORT_BY_DEVICE, numBits, blockSize);
    #if DEBUG
    printArray(output, n);
    #endif
    checkCorrectness(output, correctOutput, n);

    free(input);
    free(output);
    free(correctOutput);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
