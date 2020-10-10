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

void sortByHost(const uint32_t * in, int n, uint32_t * out, int nBits) {
    int nBins = 1 << nBits; // 2^nBits

    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bit)
	// In each loop, sort elements according to the current digit from src to dst 
	// (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
    	// TODO: Compute histogram
        memset(hist, 0, nBins * sizeof(int));
        for (int i = 0; i < n; i++) {
            int bin = (src[i] >> bit) & (nBins - 1);
            hist[bin]++;
        }

    	// TODO: Scan histogram (exclusively)
        histScan[0] = 0;
        for (int bin = 1; bin < nBins; bin++) {
            histScan[bin] = histScan[bin - 1] + hist[bin - 1];
        }

    	// TODO: Scatter elements to correct locations
        for (int i = 0; i < n; i++) {
            int bin = (src[i] >> bit) & (nBins - 1);
            dst[histScan[bin]] = src[i];
            histScan[bin]++;
        }
    	
    	// Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Copy result to out
    memcpy(out, src, n * sizeof(uint32_t)); 

    // Free memory
    free(originalSrc);
    free(hist);
    free(histScan);
}

void sortByThrust(const uint32_t * in, int n, uint32_t * out) {
    thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}

void sortByDevice(const uint32_t * in, int n, uint32_t * out, int nBits, int blockSize)
{
    // TODO
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
