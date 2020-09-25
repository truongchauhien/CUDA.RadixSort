#include "common/common.h"
#include <stdio.h>
#include <stdlib.h>

#define DEBUG 0
#define DEVICE_ID 0

typedef enum { BY_HOST, BY_DEVICE } Implementation;

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

__host__ void scanByDevice(const int *input, int n, int *output, int blockSize) {
    dim3 blockDim(blockSize);
    dim3 gridDim((n - 1) / blockDim.x + 1);

    int *d_input = NULL, *d_output = NULL;
    size_t memSize = n * sizeof(int);
    CHECK(cudaMalloc(&d_input, memSize));
    CHECK(cudaMalloc(&d_output, memSize));
    CHECK(cudaMemcpy(d_input, input, memSize, cudaMemcpyHostToDevice));

    int *d_blockSums = NULL;
    int *blockSums = NULL;
    size_t auxMemSize = 0;
    if (gridDim.x > 1) {
        auxMemSize = gridDim.x * sizeof(int);
        blockSums = (int *)malloc(auxMemSize);
        CHECK(cudaMalloc(&d_blockSums, auxMemSize));
    }

    size_t smemSize = blockDim.x * sizeof(int);
    scanPerBlock<<<gridDim, blockDim, smemSize>>>(d_input, n, d_output, d_blockSums);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    if (gridDim.x > 1) {
        cudaMemcpy(blockSums, d_blockSums, auxMemSize, cudaMemcpyDeviceToHost);
        for (int index = 1; index < gridDim.x; ++index) {
            blockSums[index] += blockSums[index - 1];
        }
        cudaMemcpy(d_blockSums, blockSums, auxMemSize, cudaMemcpyHostToDevice);

        dim3 addScannedBlockSumsGridDim(gridDim.x - 1);
        addScannedBlockSumsToScannedBlocks<<<addScannedBlockSumsGridDim, blockDim>>>(d_blockSums, d_output, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(output, d_output, memSize, cudaMemcpyDeviceToHost));

    free(blockSums);
    CHECK(cudaFree(d_blockSums));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_input));
}

__host__ void scanByHost(const int *input, int n, int *output) {
    output[0] = 0;
    for (int index = 1; index < n; ++index) {
        output[index] = output[index - 1] + input[index - 1];
    }
}

void scan(const int* input, int n, int *output, Implementation implementation = BY_HOST, int blockSize = 0) {
    GpuTimer timer;
    timer.Start();

    if (implementation == BY_HOST) {
        printf("[Scan by Host]\n");
        scanByHost(input, n, output);
    } else if (implementation == BY_DEVICE) {
        printf("[Scan by Device]\n");
        scanByDevice(input, n, output, blockSize);
    } else {

    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printArray(const int *input, int n) {
    for (int index = 0; index < n; ++index) {
        printf("%d\t", input[index]);
    }
    printf("\n");
}

void checkCorrectness(const int *output, const int *correctOutput, int n) {
    for (int index = 0; index < n; ++index) {
        if (output[index] != correctOutput[index]) {
            printf("INCORRECT!\n");
            return;
        }
    }
    printf("CORRECT!\n");
}

void getDeviceInformation() {
    cudaSetDevice(DEVICE_ID);
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, DEVICE_ID);
    printf("[Device #0 Information]\n");
    printf("- Device Name: %s\n", deviceProps.name);
    printf("- Computate capability: %d.%d\n", deviceProps.major, deviceProps.minor);
    printf("- Total global memory: %d (GB)\n", (int)(deviceProps.totalGlobalMem / 1024 / 1024 / 1024));
    printf("- Total constant memory: %d (KB)\n", (int)(deviceProps.totalConstMem / 1024));
    printf("- Shared memory per block: %d (KB)\n", (int)(deviceProps.sharedMemPerBlock / 1024));
    printf("- Max 32-bit registers per block: %d\n", deviceProps.regsPerBlock);
    printf("- Max threads per block: %d\n", deviceProps.maxThreadsPerBlock);
    printf("- Warp size: %d (threads)\n", deviceProps.warpSize);
    printf("\n");
}

int main(int argc, char* argv[]) {
    getDeviceInformation();

    int n = 1 << 24;
    int blockSize = 512;
    if (argc > 2) {
        blockSize = atoi(argv[1]);
    }

    if (DEBUG) {
        n = 16;
        blockSize = 8;
    }
    
    size_t memSize = n * sizeof(int);
    int *input = (int *)malloc(memSize);
    int *output = (int *)malloc(memSize);
    int *correctOutput = (int *)malloc(memSize);

    for (int index = 0; index < n; ++index) {
        input[index] = (rand() & 0b11);
    }

    if (DEBUG) {
        printArray(input, n);
    }

    // Scan by Host.
    scan(input, n, correctOutput);
    if (DEBUG) {
        printArray(correctOutput, n);
    }

    // Scan by Device.
    scan(input, n, output, BY_DEVICE, blockSize);    
    if (DEBUG) {   
        printArray(output, n);
    }
    checkCorrectness(output, correctOutput, n);

    free(correctOutput);
    free(output);
    free(input);

    return EXIT_SUCCESS;
}
