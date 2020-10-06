#include <stdio.h>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include "common/common.h"

#define DEBUG 0
#define MEASURE_PORTION_EXECUTION_TIME 1

typedef enum { SORT_BY_HOST, SORT_BY_THRUST, SORT_BY_DEVICE } Implementation;

void sortByHost(const uint32_t *input, int n, uint32_t *output, int numBits) {
    int nBins = 1 << numBits;

    size_t inputMemSize = n * sizeof(uint32_t);
    uint32_t *in = (uint32_t *)malloc(inputMemSize);
    uint32_t *originalIn = in;
    memcpy(in, input, inputMemSize);
    uint32_t *out = output;

    int *hist = (int *)malloc(nBins * sizeof(int));
    int *histScan = (int *)malloc(nBins * sizeof(int));

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += numBits) {
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

void sortByThrust(const uint32_t *input, int n, uint32_t *output) {
    thrust::device_vector<uint32_t> d_output(input, input + n);
	thrust::sort(d_output.begin(), d_output.end());
	thrust::copy(d_output.begin(), d_output.end(), output);
}

__device__ uint32_t mask(uint32_t number, int startBit, int numBits) {
    return (number >> startBit) & ((0b1 << numBits) - 1);
}

__global__ void sortLocallyDataBlocksKernel(const uint32_t *g_input, int n, uint32_t *g_output, int startBit, int numBits) {
    extern __shared__ uint32_t s_data[];
    uint32_t *s_input  = s_data;
    uint32_t *s_scan   = s_data + blockDim.x;
    uint32_t *s_output = s_data + blockDim.x * 2;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        s_input[threadIdx.x] = g_input[index];
    }
    __syncthreads();

    for (int bitOffset = 0; bitOffset < numBits; ++bitOffset) {
        if (index < n) {
            s_scan[threadIdx.x] = mask(s_input[threadIdx.x], startBit + bitOffset, 1);
        } else {
            s_scan[threadIdx.x] = 0;
        }
        __syncthreads();

        #pragma region Scan
        int offset = 1;
        for (int nNodes = blockDim.x / 2; nNodes > 0; nNodes /= 2) {
            offset *= 2;
            if (threadIdx.x < nNodes) {
                s_scan[threadIdx.x * offset + offset - 1]
                    += s_scan[threadIdx.x * offset + (offset / 2) - 1];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            s_scan[blockDim.x - 1] = 0;
        }
        __syncthreads();

        uint32_t temp;
        for (int nNodes = 1; nNodes < blockDim.x; nNodes *= 2) {
            if (threadIdx.x < nNodes) {
                temp = s_scan[threadIdx.x * offset + (offset / 2) - 1];
                s_scan[threadIdx.x * offset + (offset / 2) - 1] 
                    = s_scan[threadIdx.x * offset + offset - 1];
                s_scan[threadIdx.x * offset + offset - 1] += temp;
            }
            offset /= 2;
            __syncthreads();
        }
        #pragma endregion

        #pragma region CalculateRank_Scatter
        if (index < n) {
            int blockSize;
            if (blockIdx.x != gridDim.x - 1) {
                blockSize = blockDim.x;
            } else {
                blockSize = n - (blockIdx.x * blockDim.x);
            }

            int numZeros = blockSize - s_scan[blockSize - 1] - mask(s_input[blockSize - 1], startBit + bitOffset, 1);
            int rank;
            if (mask(s_input[threadIdx.x], startBit + bitOffset, 1) == 0) {
                rank = threadIdx.x - s_scan[threadIdx.x];
            } else {
                rank = numZeros + s_scan[threadIdx.x];
            }

            s_output[rank] = s_input[threadIdx.x];
        }
        __syncthreads();
        #pragma endregion

        uint32_t *tempPtr = s_input;
        s_input = s_output;
        s_output = tempPtr;
    }
    s_output = s_input;

    if (index < n) {
        g_output[index] = s_output[threadIdx.x];
    }
}

void sortLocallyDataBlocks(const uint32_t *input, int n, uint32_t *output, int startBit, int numBits, int blockSize) {
    uint32_t *d_input = NULL, *d_output = NULL;
    size_t inputMemSize = n * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_input, inputMemSize));
    CHECK(cudaMalloc(&d_output, inputMemSize));
    CHECK(cudaMemcpy(d_input, input, inputMemSize, cudaMemcpyHostToDevice));

    dim3 blockDim(blockSize);
    dim3 gridDim((n - 1) / blockDim.x + 1);
    size_t sharedMemSize = blockDim.x * sizeof(uint32_t) * 3;
    sortLocallyDataBlocksKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, n, d_output, startBit, numBits);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output, d_output, inputMemSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
}

__global__ void scatterKernel(const uint32_t *g_input, int n, uint32_t *g_output, uint32_t *g_histogramTableScan, int startBit, int numBits) {
    extern __shared__ uint8_t sharedMemory[];
    uint32_t *s_input = (uint32_t *)sharedMemory;
    uint32_t *s_firstIndices = (uint32_t *)(sharedMemory + blockDim.x * sizeof(uint32_t));

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        s_input[threadIdx.x] = g_input[index];
    }
    __syncthreads();

    if (threadIdx.x == 0 && index < n) {
        s_firstIndices[
            mask(s_input[0], startBit, numBits)
        ] = 0;
    }

    if (threadIdx.x > 0 && index < n) {
        uint32_t currentValue  = mask(s_input[threadIdx.x], startBit, numBits);
        uint32_t previousValue = mask(s_input[threadIdx.x - 1], startBit, numBits);
        if (currentValue != previousValue) {
            s_firstIndices[currentValue] = threadIdx.x;
        }
    }
    __syncthreads();

    if (index < n) {
        int numBins = 1 << numBits;
        uint32_t currentValue  = mask(s_input[threadIdx.x], startBit, numBits);
        uint32_t rank = g_histogramTableScan[blockIdx.x * numBins + currentValue]
                            + threadIdx.x - s_firstIndices[currentValue];
        g_output[rank] = s_input[threadIdx.x];
    }
}

void scatter(const uint32_t *input, int n, uint32_t *output, uint32_t *histogramTableScan, int startBit, int numBits, int blockSize) {
    uint32_t *d_input = NULL, *d_output = NULL;
    uint32_t *d_histogramTableScan = NULL;
    
    dim3 blockDim(blockSize);
    dim3 gridDim((n - 1) / blockDim.x + 1);
    int numBins = 1 << numBits;
    size_t histogramTableScanMemSize = gridDim.x * numBins * sizeof(uint32_t);
    size_t inputMemSize = n * sizeof(uint32_t);
    size_t sharedMemSize = blockDim.x * sizeof(uint32_t) + numBins * sizeof(uint32_t);

    CHECK(cudaMalloc(&d_input, inputMemSize));
    CHECK(cudaMalloc(&d_output, inputMemSize));
    CHECK(cudaMalloc(&d_histogramTableScan, histogramTableScanMemSize));
    CHECK(cudaMemcpy(d_input, input, inputMemSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_histogramTableScan, histogramTableScan, histogramTableScanMemSize, cudaMemcpyHostToDevice));
    
    scatterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, n, d_output, d_histogramTableScan, startBit, numBits);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output, d_output, inputMemSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_histogramTableScan));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_input));
}

__global__ void histogramKernel(const uint32_t *g_input, int n, uint32_t *g_histogramTable, int startBit, int numBits) {
    extern __shared__ uint32_t s_localHistogram[];
    
    size_t numBins = 0b1 << numBits;
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        s_localHistogram[bin] = 0;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        int bin = mask(g_input[index], startBit, numBits);
        atomicAdd(&s_localHistogram[bin], 1);
    }
    __syncthreads();
    
    uint32_t *localHistogram = g_histogramTable + blockIdx.x * numBins;
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        atomicAdd(&localHistogram[bin], s_localHistogram[bin]);
    }
}

void histogram(const uint32_t *input, int n, uint32_t *histogramTable, int startBit, int numBits, int blockSize) {
    uint32_t *d_input = NULL;
    uint32_t *d_histogramTable = NULL;
    
    dim3 blockDim(blockSize);
    dim3 gridDim((n - 1) / blockDim.x + 1);

    size_t inputMemSize = n * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_input, inputMemSize));
    CHECK(cudaMemcpy(d_input, input, inputMemSize, cudaMemcpyHostToDevice));
    
    size_t numBins = 0b1 << numBits;
    size_t histogramTableMemSize = gridDim.x * numBins * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_histogramTable, histogramTableMemSize));
    CHECK(cudaMemset(d_histogramTable, 0, histogramTableMemSize));

    size_t sharedMemSize = numBins * sizeof(uint32_t);
    histogramKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, n, d_histogramTable, startBit, numBits);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(histogramTable, d_histogramTable, histogramTableMemSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_histogramTable));
    CHECK(cudaFree(d_input));
}

void sortByDevice(const uint32_t *input, int n, uint32_t *output, int numBits, int blockSize) {
    // ============================================
    #if MEASURE_PORTION_EXECUTION_TIME
    GpuTimer timer;
    float histogramElapsedTime = 0.0f;
    float scanElapsedTime = 0.0f;
    float sortingLocallyBlocksElapsedTime = 0.0f;
    float scatteringElapsedTime = 0.0f;
    #endif
    // ============================================

    int numBlocks = (n - 1) / blockSize + 1;
    int numBins = 1 << numBits;
    
    size_t inputMemSize = n * sizeof(uint32_t);
    uint32_t *source      = (uint32_t *)malloc(inputMemSize);
    uint32_t *destination = (uint32_t *)malloc(inputMemSize);
    memcpy(source, input, inputMemSize);

    size_t histogramTableMemSize = (numBlocks * numBins) * sizeof(uint32_t);
    uint32_t *histogramTable     = (uint32_t *)malloc(histogramTableMemSize);
    uint32_t *histogramTableScan = (uint32_t *)malloc(histogramTableMemSize);

    for (int startBit = 0; startBit < sizeof(uint32_t) * 8; startBit += numBits) {
        // ============================================
        // Calculate local histogram for each block.
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Start();
        #endif


        // memset(histogramTable, 0, histogramTableMemSize);
        // for (int blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
        //     uint32_t *blockPtr = source + (blockIndex * blockSize);
        //     uint32_t *blockHistogram = histogramTable + (blockIndex * numBins);
        //     for (int localIndex = 0; localIndex < blockSize; ++localIndex) {
        //         int index = localIndex + blockIndex * blockSize;
        //         if (index < n) {
        //             int bin = (blockPtr[localIndex] >> startBit) & (numBins - 1);
        //             blockHistogram[bin]++;
        //         }
        //     }
        // }
        histogram(source, n, histogramTable, startBit, numBits, blockSize);

        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Stop();
        histogramElapsedTime += timer.Elapsed();
        #endif
        // ============================================

        // ============================================
        // Scan histograms by column-major order.
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Start();
        #endif

        histogramTableScan[0] = 0;
        for (int index = 1; index < numBins * numBlocks; ++index) {
            int currentBin = index / numBlocks;
            int currentBlockIndex = index % numBlocks;

            int previousBin = (index - 1) / numBlocks;
            int previousBlockIndex = (index - 1) % numBlocks;

            histogramTableScan[currentBlockIndex * numBins + currentBin] = 
                histogramTableScan[previousBlockIndex * numBins + previousBin]
                +   histogramTable[previousBlockIndex * numBins + previousBin];
        }

        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Stop();
        scanElapsedTime += timer.Elapsed();
        #endif
        // ============================================

        // ============================================
        // Sort internally each block.
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Start();
        #endif
        
        sortLocallyDataBlocks(source, n, source, startBit, numBits, blockSize);
        
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Stop();
        sortingLocallyBlocksElapsedTime += timer.Elapsed();
        #endif
        // ============================================

        // ============================================
        // Calculate rank and scatter.
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Start();
        #endif

        scatter(source, n, destination, histogramTableScan, startBit, numBits, blockSize);

        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Stop();
        scatteringElapsedTime += timer.Elapsed();
        #endif
        // ============================================

        uint32_t *temp = source;
        source = destination;
        destination = temp;
    }
    memcpy(output, source, inputMemSize);

    free(histogramTableScan);
    free(histogramTable);
    free(destination);
    free(source);

    #if MEASURE_PORTION_EXECUTION_TIME
    printf(">>>> Time | Histogram             : %.3f\n", histogramElapsedTime);
    printf(">>>> Time | Scan                  : %.3f\n", scanElapsedTime);
    printf(">>>> Time | Sort locally blocks   : %.3f\n", sortingLocallyBlocksElapsedTime);
    printf(">>>> Time | Scatter               : %.3f\n", scatteringElapsedTime);
    #endif
}

void sort(const uint32_t *in, int n,
          uint32_t *out,
          Implementation implementation = SORT_BY_HOST,
          int numBits = 4,
          int blockSize = 1) {
    GpuTimer timer;
    timer.Start();

    if (implementation == SORT_BY_HOST) {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out, numBits);
    } else if (implementation == SORT_BY_THRUST) {
    	printf("\nRadix Sort by Thrust library\n");
        sortByThrust(in, n, out);
    } else {
        printf("\nRadix Sort by device:\n");
        sortByDevice(in, n, out, numBits, blockSize);
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

void checkCorrectness(uint32_t *out, uint32_t *correctOut, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != correctOut[i]) {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t *a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%i ", a[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
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

    int numBits = 4;
    if (argc == 3) {
        numBits = atoi(argv[2]);
    }

    // Sorting by Host
    sort(in, n, correctOut, SORT_BY_HOST, numBits);
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
    sort(in, n, out, SORT_BY_DEVICE, numBits, blockSize);
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
