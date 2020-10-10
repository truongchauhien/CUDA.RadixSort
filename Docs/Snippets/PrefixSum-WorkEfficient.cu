#include "common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG 0
#define DEVICE_ID 0

#define NUM_BANKS 32
#define LOG2_NUM_BANKS 5

#define WITH_BANK_OFFSET(n) ((n) + ((n) >> LOG2_NUM_BANKS))

typedef enum { BY_HOST, BY_DEVICE, BY_DEVICE_UNROLL2, BY_DEVICE_UNROLL2_PAD } Implementation;

__global__ void scanBlocks(const int *input, int numElements, int *output, int *blockSums) {
    // SMEM Size: blockDim.x elements.
    extern __shared__ int s_data[];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Load data handled by this block into SMEM.
    if (index < numElements) {
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
    if (index < numElements) {
        output[index] = s_data[threadIdx.x];
    }
}

__global__ void addScannedBlockSumsToScannedBlocks(int *blockSums, int *blockScans, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    index += blockDim.x; // Shift to right 1 data block.
    if (index < n) {
        blockScans[index] += blockSums[blockIdx.x];
    }
}

/*
    Notice: The thread block size must be power of 2 and double the size of the data block.
    In the implementation that the data block size is equal the thread block size, only half of threads work.
*/
__global__ void scanBlocksUnroll2(const int *input, int numElements, int *output, int *blockSums) {
    // SMEM Size: blockDim.x * 2 elements.
    extern __shared__ int s_data[];
    
    int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 1. Load data handled by this block into SMEM.
    int indexPart1 = index;
    int indexPart2 = blockDim.x + index;
    int sIndexPart1 = threadIdx.x;
    int sIndexPart2 = blockDim.x + threadIdx.x;
    if (indexPart1 < numElements) {
        s_data[sIndexPart1] = input[indexPart1];
    } else {
        s_data[sIndexPart1] = 0;
    }

    if (indexPart2 < numElements) {
        s_data[sIndexPart2] = input[indexPart2];
    } else {
        s_data[sIndexPart2] = 0;
    }

    // 2. Do scan with data on SMEM. Implementation of Work-Efficient algorithm.
    // >>>> Up-Sweep phase.
    int offset = 1; // Distance between 2 adjacent nodes will be added together, in the current level.
    for (int nNodes = blockDim.x; nNodes > 0; nNodes >>= 1) { // nNodes in the number of parent nodes, in the upper level.
        __syncthreads();
        if (threadIdx.x < nNodes) {
            int sIndexRight = threadIdx.x * 2 * offset + offset * 2 - 1;
            int sIndexLeft  = threadIdx.x * 2 * offset + offset - 1;
            s_data[sIndexRight] += s_data[sIndexLeft];
        }
        offset *= 2;
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockSums != NULL) {
        // Copy sum of block into block sums array.
        blockSums[blockIdx.x] = s_data[blockDim.x * 2 - 1];
    }
    if (threadIdx.x == 0) {
        // Set 0 for the last element.
        s_data[blockDim.x * 2 - 1] = 0;
    }

    // >>>> Down-Sweep phase.
    for (int nNodes = 1; nNodes <= blockDim.x; nNodes *= 2) {
        __syncthreads();
        offset >>= 1;
        if (threadIdx.x < nNodes) {
            int sIndexRight = threadIdx.x * 2 * offset + offset * 2 - 1;
            int sIndexLeft  = threadIdx.x * 2 * offset + offset - 1;
            int temp = s_data[sIndexRight];
            s_data[sIndexRight] += s_data[sIndexLeft];
            s_data[sIndexLeft] = temp;
        }
    }
    __syncthreads();

    // 3. Copy back results from SMEM to GMEM.
    if (indexPart1 < numElements) {
        output[indexPart1] = s_data[sIndexPart1];
    }

    if (indexPart2 < numElements) {
        output[indexPart2] = s_data[sIndexPart2];
    }
}

__global__ void scanBlocksUnroll2Pad(const int *input, int numElements, int *output, int *blockSums) {
        // SMEM Size: blockDim.x * 2 elements.
    extern __shared__ int s_data[];
    
    int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 1. Load data handled by this block into SMEM.
    int sIndexPartA = WITH_BANK_OFFSET(threadIdx.x);
    int sIndexPartB = WITH_BANK_OFFSET(blockDim.x + threadIdx.x);

    if (index < numElements) {
        s_data[sIndexPartA] = input[index];
    } else {
        s_data[sIndexPartA] = 0;
    }
    
    if (index + blockDim.x < numElements) {
        s_data[sIndexPartB] = input[blockDim.x + index];
    } else {
        s_data[sIndexPartB] = 0;
    }

    // 2. Do scan with data on SMEM. Implementation of Work-Efficient algorithm.
    // >>>> Up-Sweep phase.
    int offset = 1; // Distance between 2 adjacent nodes will be added together, in the current level.
    for (int nNodes = blockDim.x; nNodes > 0; nNodes >>= 1) { // nNodes in the number of parent nodes, in the upper level.
        __syncthreads();
        if (threadIdx.x < nNodes) {
            int sIndexLeft  = WITH_BANK_OFFSET(threadIdx.x * 2 * offset + offset - 1);
            int sIndexRight = WITH_BANK_OFFSET(threadIdx.x * 2 * offset + offset * 2 - 1);
            s_data[sIndexRight] += s_data[sIndexLeft];
        }
        offset *= 2;
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockSums != NULL) {
        blockSums[blockIdx.x] = s_data[WITH_BANK_OFFSET(blockDim.x * 2 - 1)];
    }
    if (threadIdx.x == 0) {
        s_data[WITH_BANK_OFFSET(blockDim.x * 2 - 1)] = 0;
    }

    // >>>> Down-Sweep phase.
    for (int nNodes = 1; nNodes <= blockDim.x; nNodes *= 2) {
        __syncthreads();
        offset >>= 1;
        if (threadIdx.x < nNodes) {
            int sIndexLeft  = WITH_BANK_OFFSET(threadIdx.x * 2 * offset + offset - 1);
            int sIndexRight = WITH_BANK_OFFSET(threadIdx.x * 2 * offset + offset * 2 - 1);
            int temp = s_data[sIndexRight];
            s_data[sIndexRight] += s_data[sIndexLeft];
            s_data[sIndexLeft] = temp;
        }
    }
    __syncthreads();

    // 3. Copy back results from SMEM to GMEM.
    if (index < numElements) {
        output[index] = s_data[sIndexPartA];
    }

    if (blockDim.x + index < numElements) {
        output[blockDim.x + index] = s_data[sIndexPartB];
    }
}

__global__ void addScannedBlockSumsToScannedBlocksUnroll2(int *blockSums, int *blockScans, int n) {
    int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    index += blockDim.x * 2; // Shift to right 1 data block.
    if (index < n) {
        blockScans[index] += blockSums[blockIdx.x];
    }
    if (blockDim.x + index < n) {
        blockScans[blockDim.x + index] += blockSums[blockIdx.x];
    }
}

__host__ void scanByDevice(const int *input, int numElements, int *output, 
                           Implementation implementation = BY_DEVICE_UNROLL2, int blockSize = 512) {
    int *d_input = NULL, *d_output = NULL;
    size_t inpuMemSize = numElements * sizeof(int);
    CHECK(cudaMalloc(&d_input, inpuMemSize));
    CHECK(cudaMalloc(&d_output, inpuMemSize));
    CHECK(cudaMemcpy(d_input, input, inpuMemSize, cudaMemcpyHostToDevice));

    dim3 blockDim(blockSize);
    dim3 gridDim;
    size_t sharedMemSize;
    switch (implementation) {
        case BY_DEVICE:
            gridDim.x = (numElements - 1) / blockDim.x + 1;
            sharedMemSize = blockDim.x * sizeof(int);
        case BY_DEVICE_UNROLL2:
            gridDim.x = (numElements - 1) / (blockDim.x * 2) + 1;
            sharedMemSize = blockDim.x * 2 * sizeof(int);
            break;
        case BY_DEVICE_UNROLL2_PAD:
            gridDim.x = (numElements - 1) / (blockDim.x * 2) + 1;
            sharedMemSize = WITH_BANK_OFFSET(blockDim.x * 2) * sizeof(int);
            break;
        default:
            break;
    }

    int *d_blockSums = NULL, *h_blockSums = NULL;
    size_t blockSumsMemSize = 0;
    if (gridDim.x > 1) {
        blockSumsMemSize = gridDim.x * sizeof(int);
        h_blockSums = (int *)malloc(blockSumsMemSize);
        CHECK(cudaMalloc(&d_blockSums, blockSumsMemSize));
    }

    switch (implementation) {
        case BY_DEVICE:
            scanBlocks<<<gridDim, blockDim, sharedMemSize>>>(d_input, numElements, d_output, d_blockSums);
            break;
        case BY_DEVICE_UNROLL2:
            scanBlocksUnroll2<<<gridDim, blockDim, sharedMemSize>>>(d_input, numElements, d_output, d_blockSums);
            break;
        case BY_DEVICE_UNROLL2_PAD:
            scanBlocksUnroll2Pad<<<gridDim, blockDim, sharedMemSize>>>(d_input, numElements, d_output, d_blockSums);
            break;
        default:
            break;
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    if (gridDim.x > 1) {
        cudaMemcpy(h_blockSums, d_blockSums, blockSumsMemSize, cudaMemcpyDeviceToHost);
        for (int index = 1; index < gridDim.x; ++index) {
            h_blockSums[index] += h_blockSums[index - 1];
        }
        cudaMemcpy(d_blockSums, h_blockSums, blockSumsMemSize, cudaMemcpyHostToDevice);

        switch (implementation) {
            case BY_DEVICE_UNROLL2:
            case BY_DEVICE_UNROLL2_PAD:
                addScannedBlockSumsToScannedBlocksUnroll2<<<
                    dim3(gridDim.x - 1),
                    blockDim
                >>>(d_blockSums, d_output, numElements);
                break;
            default:
                addScannedBlockSumsToScannedBlocks<<<
                    dim3(gridDim.x - 1),
                    blockDim
                >>>(d_blockSums, d_output, numElements);
        }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(output, d_output, inpuMemSize, cudaMemcpyDeviceToHost));

    free(h_blockSums);
    CHECK(cudaFree(d_blockSums));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_input));
}

__host__ void scanByHost(const int *input, int numElements, int *output) {
    output[0] = 0;
    for (int index = 1; index < numElements; ++index) {
        output[index] = output[index - 1] + input[index - 1];
    }
}

void scan(const int* input, int numElements, int *output, Implementation implementation = BY_HOST, int blockSize = 0) {
    GpuTimer timer;
    timer.Start();

    if (implementation == BY_HOST) {
        printf("[Scan by Host]\n");
        scanByHost(input, numElements, output);
    } else if (implementation == BY_DEVICE) {
        printf("[Scan by Device]\n");
        scanByDevice(input, numElements, output, BY_DEVICE, blockSize);
    } else if (implementation == BY_DEVICE_UNROLL2) {
        printf("[Scan by Device (unroll 2)]\n");
        scanByDevice(input, numElements, output, BY_DEVICE_UNROLL2, blockSize);
    } else if (implementation == BY_DEVICE_UNROLL2_PAD) {
        printf("[Scan by Device (unroll 2, pad)]\n");
        scanByDevice(input, numElements, output, BY_DEVICE_UNROLL2_PAD, blockSize);
    } else {
        printf("[Unknown Implementation!]\n");
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printArray(const int *input, int numElements) {
    for (int index = 0; index < numElements; ++index) {
        printf("%d\t", input[index]);
    }
    printf("\n");
}

void checkCorrectness(const int *output, const int *correctOutput, int numElements) {
    for (int index = 0; index < numElements; ++index) {
        if (output[index] != correctOutput[index]) {
            printf("INCORRECT!\n");
            return;
        }
    }
    printf("CORRECT!\n");
}

void getDeviceInformation() {
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
    cudaSetDevice(DEVICE_ID);
    getDeviceInformation();

    int numElements = 1 << 24;
    int blockSize = 512;
    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }

    #if DEBUG
    numElements = 16;
    blockSize = 4;
    #endif
    
    printf("Number of elements: %d\n", numElements);
    printf("Block size: %d\n\n", blockSize);

    size_t inputMemSize = numElements * sizeof(int);
    int *input = (int *)malloc(inputMemSize);
    int *output = (int *)malloc(inputMemSize);
    int *correctOutput = (int *)malloc(inputMemSize);

    for (int index = 0; index < numElements; ++index) {
        input[index] = (rand() & 0b11);
    }
    #if DEBUG
    printArray(input, numElements);
    #endif

    // Scan by Host.
    scan(input, numElements, correctOutput);
    #if DEBUG
    printArray(correctOutput, numElements);
    #endif
    printf("\n");

    // Scan by Device.
    scan(input, numElements, output, BY_DEVICE, blockSize);
    checkCorrectness(output, correctOutput, numElements);
    #if DEBUG
    printArray(output, numElements);
    #endif
    printf("\n");

    // Scan by Device (unroll 2).
    memset(output, 0, inputMemSize);
    scan(input, numElements, output, BY_DEVICE_UNROLL2, blockSize);
    checkCorrectness(output, correctOutput, numElements);
    #if DEBUG
    printArray(output, numElements);
    #endif
    printf("\n");

    // Scan by Device (unroll 2, pad).
    memset(output, 0, inputMemSize);
    scan(input, numElements, output, BY_DEVICE_UNROLL2_PAD, blockSize);
    checkCorrectness(output, correctOutput, numElements);
    #if DEBUG
    printArray(output, numElements);
    #endif
    printf("\n");

    free(correctOutput);
    free(output);
    free(input);

    return EXIT_SUCCESS;
}
