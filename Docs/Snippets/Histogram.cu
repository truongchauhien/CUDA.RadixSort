#include "common/common.h"
#include <stdint.h>
#include <stdio.h>

typedef enum {
    BY_HOST,
    BY_DEVICE
} Implementation;

void histogramByHost(const int *input, int n, int *output, int numBins) {
    memset(output, 0, numBins * sizeof(int));
    for (int index = 0; index < n; ++index) {
        ++output[input[index]];
    }
}

__global__ void histogramByDevice(const int *input, int n, int *output, int numBins) {
    extern __shared__ int s_hist[];
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        s_hist[bin] = 0;
    }
    __syncthreads();

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        atomicAdd(&s_hist[input[index]], 1);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        atomicAdd(&output[bin], s_hist[bin]);
    }
}

void histogram(const int *input, int n, int *output, int numBins, Implementation implementation = BY_HOST, int blockSize = 512) {
    GpuTimer timer;
    timer.Start();
    
    if (implementation == BY_HOST) {
        histogramByHost(input, n, output, numBins);
    } else {
        dim3 blockDim(blockSize);
        dim3 gridDim((n - 1) / blockDim.x + 1);
        size_t sharedMemSize = numBins * sizeof(int);

        size_t inputMemSize = n * sizeof(int);
        size_t outputMemSize = numBins * sizeof(int);

        int *d_input = NULL;
        int *d_output = NULL;
        CHECK(cudaMalloc(&d_input, inputMemSize));
        CHECK(cudaMalloc(&d_output, outputMemSize));

        CHECK(cudaMemcpy(d_input, input, inputMemSize, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_output, 0, outputMemSize));
        histogramByDevice<<<gridDim, blockDim, sharedMemSize>>>(d_input, n, d_output, numBins);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());        
        CHECK(cudaMemcpy(output, d_output, outputMemSize, cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_output));
        CHECK(cudaFree(d_input));
    }

    timer.Stop();
    printf("Time: %.3f\n", timer.Elapsed());
}

void checkCorrectness(const int *histogramByHost, const int *histogramByDevice, int numBins) {
    bool isMatched = true;
    for (int bin = 0; bin < numBins; ++bin) {
        if (histogramByHost[bin] != histogramByDevice[bin]) {
            isMatched = false;
            break;
        }
    }

    if (isMatched) {
        printf("CORRECT!\n");
    } else {
        printf("INCORRECT!\n");
    }
}

int main(int argc, char *argv[]) {
    int n = 1 << 24;
    int numBits = 4;
    int blockSize = 512;

    int numBins = 1 << numBits;    
    
    int *input = (int *)malloc(n * sizeof(int));
    int *correctOutput = (int *)malloc(numBins * sizeof(int));
    int *output = (int *)malloc(numBins * sizeof(int));
    
    for (int index = 0; index < n; ++index) {
        input[index] = rand() & (numBins - 1);
    }

    printf("By Host\n");
    histogram(input, n, correctOutput, numBins);
    printf("\n");

    printf("By Device\n");
    histogram(input, n, output, numBins, BY_DEVICE, blockSize);
    checkCorrectness(correctOutput, output, numBins);
    printf("\n");

    free(output);
    free(correctOutput);
    free(input);

    return EXIT_SUCCESS;
}
