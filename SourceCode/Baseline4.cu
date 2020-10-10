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
#define MEASURE_PORTION_EXECUTION_TIME 1

typedef enum { SORT_BY_HOST, SORT_BY_THRUST, SORT_SEQUENTIALLY_BY_HOST_USING_PARALLEL_ALGORITHM } Implementation;

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

void sortByThrust(const uint32_t *in, int n, uint32_t *out) {
    thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}

void sortByHostUsingParallelAlgorithm(const uint32_t *input, int n, uint32_t *output, int numBits, int blockSize) {
    // ============================================
    #if MEASURE_PORTION_EXECUTION_TIME
    GpuTimer timer;
    float histogramElapsedTime = 0.0f;
    float scanElapsedTime = 0.0f;
    float internalBlockSortingElapsedTime = 0.0f;
    float firstIndicesFindingElapedTime = 0.0f;
    float scatteringElapsedTime = 0.0f;
    #endif
    // ============================================

    int numBlocks = (n - 1) / blockSize + 1;
    int numBins = 1 << numBits;
    
    size_t inputMemSize = n * sizeof(uint32_t);
    uint32_t *source      = (uint32_t *)malloc(inputMemSize);
    uint32_t *destination = (uint32_t *)malloc(inputMemSize);
    memcpy(source, input, inputMemSize);

    size_t histogramBlocksMemSize = (numBlocks * numBins) * sizeof(int);
    int *blockHistograms     = (int *)malloc(histogramBlocksMemSize);
    int *blockHistogramsScan = (int *)malloc(histogramBlocksMemSize);

    int      *block1BitScan = (int *)     malloc(blockSize * sizeof(int));
    uint32_t *sortedBlock   = (uint32_t *)malloc(blockSize * sizeof(uint32_t));
    int      *firstIndices  = (int *)     malloc(numBins   * sizeof(int));

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += numBits) {
        // ============================================
        // Calculate local histogram for each block.
        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Start();
        #endif

        memset(blockHistograms, 0, histogramBlocksMemSize);
        for (int blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
            uint32_t *blockPtr = source + (blockIndex * blockSize);
            int *blockHistogram = blockHistograms + (blockIndex * numBins);
            for (int localIndex = 0; localIndex < blockSize; ++localIndex) {
                int index = localIndex + blockIndex * blockSize;
                if (index < n) {
                    int bin = (blockPtr[localIndex] >> bit) & (numBins - 1);
                    blockHistogram[bin]++;
                }
            }
        }

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

        blockHistogramsScan[0] = 0;
        for (int index = 1; index < numBins * numBlocks; ++index) {
            int currentBin = index / numBlocks;
            int currentBlockIndex = index % numBlocks;

            int previousBin = (index - 1) / numBlocks;
            int previousBlockIndex = (index - 1) % numBlocks;

            blockHistogramsScan[currentBlockIndex * numBins + currentBin] = 
                blockHistogramsScan[previousBlockIndex * numBins + previousBin]
                +   blockHistograms[previousBlockIndex * numBins + previousBin];
        }

        #if MEASURE_PORTION_EXECUTION_TIME
        timer.Stop();
        scanElapsedTime += timer.Elapsed();
        #endif
        // ============================================

        // ============================================
        // Sort internally each block, then calculate rank and scatter.
        for (int blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
            uint32_t *blockPtr = source + (blockIndex * blockSize);

            // ============================================
            #if MEASURE_PORTION_EXECUTION_TIME
            timer.Start();
            #endif

            for (int innerBit = 0; innerBit < numBits; ++innerBit) {
                block1BitScan[0] = 0;
                for (int localIndex = 1; localIndex < blockSize; ++localIndex) {
                    int index = localIndex + blockIndex * blockSize;
                    int inputValue = 0;
                    if (index < n) {
                        inputValue = (blockPtr[localIndex - 1] >> (bit + innerBit)) & 0b1;
                    }
                    block1BitScan[localIndex] = 
                        block1BitScan[localIndex - 1] + inputValue;
                }

                int numZeros;
                if (blockIndex != numBlocks - 1) {
                    numZeros = blockSize
                        - block1BitScan[blockSize - 1]
                        - ((blockPtr[blockSize - 1] >> (bit + innerBit)) & 0b1);
                } else {
                    int realBlockSize = n - (blockIndex * blockSize);
                    numZeros = realBlockSize
                        - block1BitScan[realBlockSize - 1]
                        - ((blockPtr[realBlockSize - 1] >> (bit + innerBit)) & 0b1);
                }
                for (int localIndex = 0; localIndex < blockSize; ++localIndex) {
                    int index = localIndex + blockIndex * blockSize;
                    if (index < n) {
                        int bitValue = (blockPtr[localIndex] >> (bit + innerBit)) & 0b1;
                        int rank;
                        if (bitValue == 0) {
                            rank = localIndex - block1BitScan[localIndex];
                        } else {
                            rank = numZeros + block1BitScan[localIndex];
                        }
                        sortedBlock[rank] = blockPtr[localIndex];
                    }
                }
                for (int localIndex = 0; localIndex < blockSize; ++localIndex) {
                    int index = localIndex + blockIndex * blockSize;
                    if (index < n) {
                        blockPtr[localIndex] = sortedBlock[localIndex];
                    }
                }
            }

            #if MEASURE_PORTION_EXECUTION_TIME
            timer.Stop();
            internalBlockSortingElapsedTime += timer.Elapsed();
            #endif
            // ============================================

            // ============================================
            #if MEASURE_PORTION_EXECUTION_TIME
            timer.Start();
            #endif

            firstIndices[(blockPtr[0] >> bit) & (numBins - 1)] = 0;
            for (int localIndex = 1; localIndex < blockSize; ++localIndex) {
                int index = localIndex + blockIndex * blockSize;
                if (index < n) {
                    int currentValue  = (blockPtr[localIndex]     >> bit) & (numBins - 1);
                    int previousValue = (blockPtr[localIndex - 1] >> bit) & (numBins - 1);
                    if (currentValue != previousValue) {
                        firstIndices[currentValue] = localIndex;
                    }
                }
            }

            #if MEASURE_PORTION_EXECUTION_TIME
            timer.Stop();
            firstIndicesFindingElapedTime += timer.Elapsed();
            #endif
            // ============================================

            // ============================================
            #if MEASURE_PORTION_EXECUTION_TIME
            timer.Start();
            #endif

            for (int localIndex = 0; localIndex < blockSize; ++localIndex) {
                int index = localIndex + blockIndex * blockSize;
                if (index < n) {
                    int currentValue = (blockPtr[localIndex] >> bit) & (numBins - 1);
                    int rank = blockHistogramsScan[blockIndex * numBins + currentValue]
                               + localIndex - firstIndices[currentValue];
                    destination[rank] = blockPtr[localIndex];
                }
            }

            #if MEASURE_PORTION_EXECUTION_TIME
            timer.Stop();
            scatteringElapsedTime += timer.Elapsed();
            #endif
            // ============================================
        }
        // ============================================

        uint32_t *temp = source;
        source = destination;
        destination = temp;
    }
    memcpy(output, source, inputMemSize);

    free(firstIndices);
    free(sortedBlock);
    free(block1BitScan);
    free(blockHistogramsScan);
    free(blockHistograms);
    free(destination);
    free(source);

    #if MEASURE_PORTION_EXECUTION_TIME
    printf(">>>> Time | Histogram             : %.3f\n", histogramElapsedTime);
    printf(">>>> Time | Scan                  : %.3f\n", scanElapsedTime);
    printf(">>>> Time | Sort internally block : %.3f\n", internalBlockSortingElapsedTime);
    printf(">>>> Time | Find First Indices    : %.3f\n", firstIndicesFindingElapedTime);
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
        printf("\nRadix Sort by host, using parallel algorithm\n");
        sortByHostUsingParallelAlgorithm(in, n, out, numBits, blockSize);
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

    // Sorting by Host, using parallel algorithm designed for Device.
    memset(output, 0u, inputMemSize);
    sort(input, n, output, SORT_SEQUENTIALLY_BY_HOST_USING_PARALLEL_ALGORITHM, numBits, blockSize);
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
