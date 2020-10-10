#include <stdio.h>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include "common/common.h"

#define DEVICE_ID 0
#define DEBUG 0
#define ENABLE_CHECK_ERROR 0
#define MEASURE_PORTION_EXECUTION_TIME 1

#if !ENABLE_CHECK_ERROR
    #undef CHECK // Remove CHECK macro.
    #define CHECK
#endif

#define TRANSPOSE_SMEM_PAD 2

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

__device__ __forceinline__ uint32_t mask(uint32_t number, int startBit, int numBits) {
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

void sortLocallyDataBlocks(const uint32_t *d_input, int n, uint32_t *d_output, int startBit, int numBits, int blockSize) {
    dim3 blockDim(blockSize);
    dim3 gridDim((n - 1) / blockDim.x + 1);
    size_t sharedMemSize = blockDim.x * sizeof(uint32_t) * 3;
    sortLocallyDataBlocksKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, n, d_output, startBit, numBits);
    #if ENABLE_CHECK_ERROR
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    #endif
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

void scatter(const uint32_t *d_input, int n, uint32_t *d_output, uint32_t *d_histogramTableScan, int startBit, int numBits, int blockSize) {   
    dim3 blockDim(blockSize);
    dim3 gridDim((n - 1) / blockDim.x + 1);
    int numBins = 1 << numBits;
    size_t sharedMemSize = blockDim.x * sizeof(uint32_t) + numBins * sizeof(uint32_t);    
    scatterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, n, d_output, d_histogramTableScan, startBit, numBits);
    #if ENABLE_CHECK_ERROR
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    #endif
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

void histogram(const uint32_t *d_input, int n, uint32_t *d_histogramTable, int startBit, int numBits, int blockSize) {
    dim3 blockDim(blockSize);
    dim3 gridDim((n - 1) / blockDim.x + 1);
    
    size_t numBins = 0b1 << numBits;
    size_t histogramTableMemSize = gridDim.x * numBins * sizeof(uint32_t);
    CHECK(cudaMemset(d_histogramTable, 0, histogramTableMemSize));

    size_t sharedMemSize = numBins * sizeof(uint32_t);
    histogramKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, n, d_histogramTable, startBit, numBits);
    #if ENABLE_CHECK_ERROR
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    #endif
}

__global__ void transposedKernel(const uint32_t *g_input, int numCols, int numRows, uint32_t *g_output) {
    extern __shared__ uint32_t s_tile[];

    int inCol = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inIndex = inRow * numCols + inCol;
    
    int insideBlockIndex = threadIdx.y * (blockDim.x * 2 + TRANSPOSE_SMEM_PAD) + threadIdx.x;
    int insideBlockIndexWithoutPad = threadIdx.y * blockDim.x + threadIdx.x;
    int insideTransposedBlockCol = insideBlockIndexWithoutPad % blockDim.y;
    int insideTransposedBlockRow = insideBlockIndexWithoutPad / blockDim.y;
    int insideTransposedBlockIndex = insideTransposedBlockCol * (blockDim.x * 2 + TRANSPOSE_SMEM_PAD) + insideTransposedBlockRow;

    int outCol = blockIdx.y * blockDim.y + insideTransposedBlockCol;
    int outRow = blockIdx.x * (blockDim.x * 2) + insideTransposedBlockRow;
    int outIndex = outRow * numRows + outCol;

    if (inCol < numCols && inRow < numRows) {
        s_tile[insideBlockIndex] = g_input[inIndex];
    }
    if (inCol + blockDim.x < numCols && inRow < numRows) {
        s_tile[insideBlockIndex + blockDim.x] = g_input[inIndex + blockDim.x];
    }
    __syncthreads();

    if (outCol < numRows && outRow < numCols) {
        g_output[outIndex] = s_tile[insideTransposedBlockIndex];
    }
    if (outCol < numRows && outRow + blockDim.x < numCols) {
        g_output[outIndex + numRows * blockDim.x] = s_tile[insideTransposedBlockIndex + blockDim.x];
    }
}

void transpose(const uint32_t *d_input, int numCols, int numRows, uint32_t *d_output) {
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (numCols - 1) / (blockDim.x * 2) + 1,
        (numRows - 1) / blockDim.y + 1
    );
    size_t sharedMemSize = blockDim.y * (blockDim.x * 2 + TRANSPOSE_SMEM_PAD) * sizeof(uint32_t);
    transposedKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, numCols, numRows, d_output);
    #if ENABLE_CHECK_ERROR
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    #endif
}

__global__ void scanBlocks(const uint32_t *input, int n, uint32_t *output, uint32_t *blockSums) {
    // SMEM Size: blockDim.x elements.
    extern __shared__ uint32_t s_data[];
    
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

__global__ void addScannedBlockSumsToScannedBlocks(uint32_t *blockSums, uint32_t *blockScans, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    index += blockDim.x; // Shift to right 1 block.
    if (index < n) {
        blockScans[index] += blockSums[blockIdx.x];
    }
}

void scan(const uint32_t *d_input, int n, uint32_t *d_output, int blockSize) {
    dim3 blockDim(blockSize);
    dim3 gridDim((n - 1) / blockDim.x + 1);
    
    static uint32_t *d_blockSums = NULL;
    static uint32_t *h_blockSums = NULL;
    static size_t blockSumsMemSize = 0;

    size_t newBlockSumsMemSize = gridDim.x * sizeof(uint32_t);
    if (gridDim.x > 1 && blockSumsMemSize != newBlockSumsMemSize)  {
        if (d_blockSums != NULL) {
            CHECK(cudaFree(d_blockSums));
        }
        if (h_blockSums != NULL) {
            delete[] h_blockSums;
        }

        blockSumsMemSize = newBlockSumsMemSize;
        h_blockSums = new uint32_t[gridDim.x];
        CHECK(cudaMalloc(&d_blockSums, blockSumsMemSize));
    }

    size_t sharedMemSize = blockDim.x * sizeof(uint32_t);
    scanBlocks<<<gridDim, blockDim, sharedMemSize>>>(d_input, n, d_output, d_blockSums);
    #if ENABLE_CHECK_ERROR
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    #endif

    if (gridDim.x > 1) {
        cudaMemcpy(h_blockSums, d_blockSums, blockSumsMemSize, cudaMemcpyDeviceToHost);
        for (int index = 1; index < gridDim.x; ++index) {
            h_blockSums[index] += h_blockSums[index - 1];
        }
        cudaMemcpy(d_blockSums, h_blockSums, blockSumsMemSize, cudaMemcpyHostToDevice);

        dim3 postScanBlocksGridDim(gridDim.x - 1);
        addScannedBlockSumsToScannedBlocks<<<postScanBlocksGridDim, blockDim>>>(d_blockSums, d_output, n);
        #if ENABLE_CHECK_ERROR
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        #endif
    }
}

void sortByDevice(const uint32_t *h_input, int n, uint32_t *h_output, int numBits, int blockSize) {
    // ============================================
    #if MEASURE_PORTION_EXECUTION_TIME
    GpuTimer timer;
    float sortingLocallyBlocksElapsedTime = 0.0f;
    float histogramElapsedTime = 0.0f;
    float scanElapsedTime = 0.0f;
    float scatteringElapsedTime = 0.0f;
    #endif
    // ============================================

    int numBlocks = (n - 1) / blockSize + 1;
    int numBins = 1 << numBits;
    
    uint32_t *d_input  = NULL;
    uint32_t *d_output = NULL;
    size_t inputMemSize = n * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_input, inputMemSize));
    CHECK(cudaMalloc(&d_output, inputMemSize));
    CHECK(cudaMemcpy(d_input, h_input, inputMemSize, cudaMemcpyHostToDevice));

    uint32_t *d_histogramTable              = NULL;
    uint32_t *d_histogramTableTranspose     = NULL;
    uint32_t *d_histogramTableScanTranspose = NULL;
    uint32_t *d_histogramTableScan          = NULL;
    size_t histogramTableMemSize = (numBlocks * numBins) * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_histogramTable, histogramTableMemSize));
    CHECK(cudaMalloc(&d_histogramTableTranspose, histogramTableMemSize));
    CHECK(cudaMalloc(&d_histogramTableScanTranspose, histogramTableMemSize));
    CHECK(cudaMalloc(&d_histogramTableScan, histogramTableMemSize));

    for (int startBit = 0; startBit < sizeof(uint32_t) * 8; startBit += numBits) {
        // ============================================
        // Sort internally each block.
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Start();
        #endif
        
        sortLocallyDataBlocks(d_input, n, d_input, startBit, numBits, blockSize);
        
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Stop();
        sortingLocallyBlocksElapsedTime += timer.Elapsed();
        #endif
        // ============================================

        // ============================================
        // Calculate local histogram for each block.
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Start();
        #endif

        histogram(d_input, n, d_histogramTable, startBit, numBits, blockSize);

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

        transpose(d_histogramTable, numBins, numBlocks, d_histogramTableTranspose);
        scan(d_histogramTableTranspose, numBlocks * numBins, d_histogramTableScanTranspose, blockSize);
        transpose(d_histogramTableScanTranspose, numBlocks, numBins, d_histogramTableScan);
        
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Stop();
        scanElapsedTime += timer.Elapsed();
        #endif
        // ============================================

        // ============================================
        // Calculate rank and scatter.
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Start();
        #endif

        scatter(d_input, n, d_output, d_histogramTableScan, startBit, numBits, blockSize);

        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Stop();
        scatteringElapsedTime += timer.Elapsed();
        #endif
        // ============================================

        uint32_t *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    CHECK(cudaMemcpy(h_output, d_input, inputMemSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_histogramTableScan));
    CHECK(cudaFree(d_histogramTableScanTranspose));
    CHECK(cudaFree(d_histogramTableTranspose));
    CHECK(cudaFree(d_histogramTable));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_input));

    #if MEASURE_PORTION_EXECUTION_TIME
    printf(">>>> Time | Sort locally blocks   : %.3f\n", sortingLocallyBlocksElapsedTime);
    printf(">>>> Time | Histogram             : %.3f\n", histogramElapsedTime);
    printf(">>>> Time | Scan                  : %.3f\n", scanElapsedTime);
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
    CHECK(cudaSetDevice(DEVICE_ID));
    CHECK(cudaDeviceReset());
    cudaFree(0);
    
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n;
    #if DEBUG
    n = 513;
    #else
    n = (1 << 24) + 1;
    #endif
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t *in = (uint32_t *)malloc(bytes);
    uint32_t *out = (uint32_t *)malloc(bytes);
    uint32_t *correctOut = (uint32_t *)malloc(bytes);

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++) {
        #if DEBUG
        in[i] = rand() & 0xFF;
        #else
        in[i] = rand();
        #endif
    }
    #if DEBUG
    printArray(in, n);
    #endif

    // DETERMINE BLOCK SIZE
    int blockSize = 512;
    if (argc == 2) {
        blockSize = atoi(argv[1]);
    }

    int numBits;
    #if DEBUG
    numBits = 4;
    #else
    numBits = 8;
    #endif
    if (argc == 3) {
        numBits = atoi(argv[2]);
    }

    // Sorting by Host
    sort(in, n, correctOut, SORT_BY_HOST, numBits);
    #if DEBUG
    printArray(correctOut, n);
    #endif

    // Sorting by Thrust Library
    sort(in, n, out, SORT_BY_THRUST);
    #if DEBUG
    printArray(out, n);
    #endif
    checkCorrectness(out, correctOut, n);
    memset(out, 0u, n * sizeof(uint32_t)); // Reset ouput.

    // Sorting by Device
    sort(in, n, out, SORT_BY_DEVICE, numBits, blockSize);
    #if DEBUG
    printArray(out, n);
    #endif
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
