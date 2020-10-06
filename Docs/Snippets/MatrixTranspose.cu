#include "common/common.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define DEBUG 0
#define DEVICE_ID 0
#define PAD 2

__host__ void transposeHost(const int *input, int numCols, int numRows, int *output) {
    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < numCols; ++col) {
            output[col * numRows + row] = input[row * numCols + col];
        }
    }
}

__global__ void transposeBlock1DGrid1D(const int *input, int numCols, int numRows, int *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int numElements = numRows * numCols;
    if (index < numElements) {
        int col = index % numCols;
        int row = index / numCols;
        output[col * numRows + row] = input[row * numCols + col];
    }
}

/*
    Read in rows, write in columns.
*/
__global__ void transposeNaiveRowToColumn(const int *input, int numCols, int numRows, int *output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < numCols && row < numRows) {
        output[col * numRows + row] = input[row * numCols + col];
    }
}

template<typename T>
__device__ void swap(T *left, T *right) {
    T temp = *left;
    *left = *right;
    *right = temp;
}

/*
    Read in colums, write in rows.
*/
__global__ void transposeNaiveColToRow(const int *input, int numCols, int numRows, int *output) {
    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inIndex = inRow * numCols + inCol;

    int outCol = blockIdx.y * blockDim.y + threadIdx.y;
    int outRow = blockIdx.x * blockDim.x + threadIdx.x;

    int outIndex = outRow * numRows + outCol;

    if (inCol < numCols && inRow < numRows) {
        output[outIndex] = input[inIndex];
    }
}

__global__ void transposeCoalesced(const int *input, int numCols, int numRows, int *output) {
    extern __shared__ int tile[];

    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inIndex = inRow * numCols + inCol;
    
    int tileIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int transposedTileIndex = threadIdx.x * blockDim.x + threadIdx.y;

    int outCol = blockIdx.y * blockDim.y + threadIdx.x;
    int outRow = blockIdx.x * blockDim.x + threadIdx.y;
    int outIndex = outRow * numRows + outCol;

    if (inCol < numCols && inRow < numRows) {
        tile[tileIndex] = input[inIndex];
    }
    __syncthreads();

    if (outCol < numRows && outRow < numCols) {
        output[outIndex] = tile[transposedTileIndex];
    }
}

__global__ void transposeCoalescedPad(const int *input, int numCols, int numRows, int *output) {
    extern __shared__ int tile[];

    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inIndex = inRow * numCols + inCol;
    
    int tileIndex = threadIdx.y * (blockDim.x + PAD) + threadIdx.x;
    int transposedTileIndex = threadIdx.x * (blockDim.x + PAD) + threadIdx.y;

    int outCol = blockIdx.y * blockDim.y + threadIdx.x;
    int outRow = blockIdx.x * blockDim.x + threadIdx.y;
    int outIndex = outRow * numRows + outCol;

    if (inCol < numCols && inRow < numRows) {
        tile[tileIndex] = input[inIndex];
    }
    __syncthreads();

    if (outCol < numRows && outRow < numCols) {
        output[outIndex] = tile[transposedTileIndex];
    }
}

/*
__global__ void transposeCoalesced(const int *input, int numCols, int numRows, int *output) {
    extern __shared__ int s_tile[];

    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inIndex = inRow * numCols + inCol;
    
    int insideBlockIndex = threadIdx.y * blockDim.x + threadIdx.x;
    int insideTransposedBlockCol = insideBlockIndex % blockDim.y;
    int insideTransposedBlockRow = insideBlockIndex / blockDim.y;
    int insideTransposedBlockIndex = insideTransposedBlockCol * blockDim.x + insideTransposedBlockRow;

    int outCol = blockIdx.y * blockDim.y + insideTransposedBlockCol;
    int outRow = blockIdx.x * blockDim.x + insideTransposedBlockRow;

    int outIndex = outRow * numRows + outCol;

    if (inCol < numCols && inRow < numRows) {
        s_tile[insideBlockIndex] = input[inIndex];
        __syncthreads();

        output[outIndex] = s_tile[insideTransposedBlockIndex];
    }
}

__global__ void transposeCoalescedConflictFree(const int *input, int numCols, int numRows, int *output) {
    extern __shared__ int s_tile[];

    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inIndex = inRow * numCols + inCol;
    
    int insideBlockIndex = threadIdx.y * (blockDim.x + PAD) + threadIdx.x;
    int insideBlockIndexWithoutPad = threadIdx.y * blockDim.x + threadIdx.x;
    int insideTransposedBlockCol = insideBlockIndexWithoutPad % blockDim.y;
    int insideTransposedBlockRow = insideBlockIndexWithoutPad / blockDim.y;
    int insideTransposedBlockIndex = insideTransposedBlockCol * (blockDim.x + PAD) + insideTransposedBlockRow;

    int outCol = blockIdx.y * blockDim.y + insideTransposedBlockCol;
    int outRow = blockIdx.x * blockDim.x + insideTransposedBlockRow;

    int outIndex = outRow * numRows + outCol;

    if (inCol < numCols && inRow < numRows) {
        s_tile[insideBlockIndex] = input[inIndex];
        __syncthreads();

        output[outIndex] = s_tile[insideTransposedBlockIndex];
    }
}
*/

__global__ void transposeCoalescedConflictFreeUnroll2(const int *input, int numCols, int numRows, int *output) {
    extern __shared__ int s_tile[];

    int inCol = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inIndex = inRow * numCols + inCol;
    
    int insideBlockIndex = threadIdx.y * (blockDim.x * 2 + PAD) + threadIdx.x;
    int insideBlockIndexWithoutPad = threadIdx.y * blockDim.x + threadIdx.x;
    int insideTransposedBlockCol = insideBlockIndexWithoutPad % blockDim.y;
    int insideTransposedBlockRow = insideBlockIndexWithoutPad / blockDim.y;
    int insideTransposedBlockIndex = insideTransposedBlockCol * (blockDim.x * 2 + PAD) + insideTransposedBlockRow;

    int outCol = blockIdx.y * blockDim.y + insideTransposedBlockCol;
    int outRow = blockIdx.x * (blockDim.x * 2) + insideTransposedBlockRow;
    int outIndex = outRow * numRows + outCol;

    if (inCol < numCols && inRow < numRows) {
        s_tile[insideBlockIndex] = input[inIndex];
    }
    if (inCol + blockDim.x && inRow < numRows) {
        s_tile[insideBlockIndex + blockDim.x] = input[inIndex + blockDim.x];
    }
    __syncthreads();

    if (outCol < numRows && outRow < numCols) {
        output[outIndex] = s_tile[insideTransposedBlockIndex];
    }
    if (outCol < numRows && outRow + blockDim.x < numCols) {
        output[outIndex + numRows * blockDim.x] = s_tile[insideTransposedBlockIndex + blockDim.x];
    }
}

__global__ void transposeCoalescedConflictFreeUnroll4(const int *input, int numCols, int numRows, int *output) {
    extern __shared__ int s_tile[];

    int inCol = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inIndex = inRow * numCols + inCol;
    
    int insideBlockIndex = threadIdx.y * (blockDim.x * 4 + PAD) + threadIdx.x;
    int insideBlockIndexWithoutPad = threadIdx.y * blockDim.x + threadIdx.x;
    int insideTransposedBlockCol = insideBlockIndexWithoutPad % blockDim.y;
    int insideTransposedBlockRow = insideBlockIndexWithoutPad / blockDim.y;
    int insideTransposedBlockIndex = insideTransposedBlockCol * (blockDim.x * 4 + PAD) + insideTransposedBlockRow;

    int outCol = blockIdx.y * blockDim.y + insideTransposedBlockCol;
    int outRow = blockIdx.x * (blockDim.x * 4) + insideTransposedBlockRow;
    int outIndex = outRow * numRows + outCol;

    if (inCol < numCols && inRow < numRows) {
        s_tile[insideBlockIndex] = input[inIndex];
    }
    if (inCol + blockDim.x < numCols && inRow < numRows) {
        s_tile[insideBlockIndex + blockDim.x] = input[inIndex + blockDim.x];
    }
    if (inCol + blockDim.x * 2 < numCols && inRow < numRows) {
        s_tile[insideBlockIndex + blockDim.x * 2] = input[inIndex + blockDim.x * 2];
    }
    if (inCol + blockDim.x * 3 < numCols && inRow < numRows) {
        s_tile[insideBlockIndex + blockDim.x * 3] = input[inIndex + blockDim.x * 3];
    }
    __syncthreads();

    if (outCol < numRows && outRow < numCols) {
        output[outIndex] = s_tile[insideTransposedBlockIndex];
    }
    if (outCol < numRows && outRow + blockDim.x < numCols) {
        output[outIndex + numRows * blockDim.x] = s_tile[insideTransposedBlockIndex + blockDim.x];
    }
    if (outCol < numRows && outRow + blockDim.x * 2 < numCols) {
        output[outIndex + numRows * blockDim.x * 2] = s_tile[insideTransposedBlockIndex + blockDim.x * 2];
    }
    if (outCol < numRows && outRow + blockDim.x * 3 < numCols) {
        output[outIndex + numRows * blockDim.x * 3] = s_tile[insideTransposedBlockIndex + blockDim.x * 3];
    }
}

void checkCorrectness(const int *hostOutput, const int *deviceOutput, int numElements) {
    for (int index = 0; index < numElements; ++index) {
        if (hostOutput[index] != deviceOutput[index]) {
            printf("INCORRECT!\n");
            return;
        }
    }

    printf("CORRECT!\n");
}

void initializeArray(int *input, int numElements) {
    for (int index = 0; index < numElements; ++index) {
        #if DEBUG
            input[index] = rand() & 0b111;
        #else
            input[index] = rand();
        #endif
    }
}

void printArray(int *input, int numCols, int numRows) {
    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < numCols; ++col) {
            printf("%d\t", input[row * numCols + col]);
        }
        if (row != numRows - 1) {
            printf("\n\n");
        }
    }
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

int main(int argc, char *argv[]) {
    getDeviceInformation();

    GpuTimer timer;

    int numRows;
    int numCols;
    #if DEBUG
    numRows = 5;
    numCols = 9;
    #else
    numRows = 4096;
    numCols = 4096;
    #endif    
    int numElements = numRows * numCols;

    int *input = new int[numElements];
    int *hostOutput = new int[numElements];
    int *deviceOutput = new int[numElements];
    initializeArray(input, numElements);

    #if DEBUG
    printf("Input matrix:\n");
    printArray(input, numCols, numRows);
    printf("\n\n");
    #endif

    dim3 blockDim1D(256);
    dim3 gridDim1D((numElements - 1) / blockDim1D.x + 1);

    #if DEBUG
    dim3 blockDimSquare(4, 4);
    dim3 blockDimRectangle(8, 4);
    #else
    dim3 blockDimSquare(32, 32);
    dim3 blockDimRectangle(64, 8);    
    #endif
    
    dim3 gridDimSquare(
        (numCols - 1) / blockDimSquare.x + 1,
        (numRows - 1) / blockDimSquare.y + 1
    );
    dim3 gridDimRectangle((numCols - 1) / blockDimRectangle.x + 1, (numRows - 1) / blockDimRectangle.y + 1);
    dim3 gridDimUnroll2(
        (numCols - 1) / (blockDimRectangle.x * 2) + 1,
        (numRows - 1) / blockDimRectangle.y + 1
    );
    dim3 gridDimUnroll4(
        (numCols - 1) / (blockDimRectangle.x * 4) + 1,
        (numRows - 1) / blockDimRectangle.y + 1
    );

    // >>>> ==================================================
    printf("Transpose by Host\n");
    timer.Start();
    transposeHost(input, numCols, numRows, hostOutput);
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());    
    #if DEBUG
    printf("Output matrix:\n");
    printArray(hostOutput, numRows, numCols);
    printf("\n\n");
    #else
    printf("\n");
    #endif
    // <<<< ==================================================

    size_t inputMemSize = numElements * sizeof(int);
    int *d_input = NULL, *d_output = NULL;
    CHECK(cudaMalloc(&d_input, inputMemSize));
    CHECK(cudaMalloc(&d_output, inputMemSize));
    CHECK(cudaMemcpy(d_input, input, inputMemSize, cudaMemcpyHostToDevice));

    // >>>> ==================================================
    printf("Transpose by Device: 1D blocks, 1D grid\n");
    
    timer.Start();
    transposeBlock1DGrid1D<<<gridDim1D, blockDim1D>>>(d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    printf("\n");
    // <<<< ==================================================

    // >>>> ==================================================
    printf("Transpose by Device: Read in rows - Write in columns\n");
    CHECK(cudaMemset(d_output, 0, inputMemSize));

    timer.Start();
    transposeNaiveRowToColumn<<<gridDimRectangle, blockDimRectangle>>>(d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    printf("\n");
    // <<<< ==================================================

    // >>>> ==================================================
    printf("Transpose by Device: Read in columns - Write in rows\n");
    CHECK(cudaMemset(d_output, 0, inputMemSize));

    timer.Start();
    transposeNaiveColToRow<<<gridDimRectangle, blockDimRectangle>>>(d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    printf("\n");
    // <<<< ==================================================

    // >>>> ==================================================
    printf("Transpose by Device: Read in rows -> SMEM (Square Tile) -> Write in rows\n");
    CHECK(cudaMemset(d_output, 0, inputMemSize));
    
    size_t sharedMemSize = blockDimSquare.x * blockDimSquare.y * sizeof(int);
    timer.Start();
    transposeCoalesced<<<
        gridDimSquare,
        blockDimSquare,
        sharedMemSize
    >>> (d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    printf("\n");
    // <<<< ==================================================

    // >>>> ==================================================
    printf("Transpose by Device: Read in rows-> SMEM (Square Tile, Padded) -> Write in rows\n");
    CHECK(cudaMemset(d_output, 0, inputMemSize));
    
    sharedMemSize = blockDimSquare.y * (blockDimSquare.x + PAD) * sizeof(int);
    timer.Start();
    transposeCoalescedPad<<<
        gridDimSquare,
        blockDimSquare,
        sharedMemSize
    >>> (d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    printf("\n");
    // <<<< ==================================================

    // >>>> ==================================================
    /*
    printf("Transpose by Device: Read in rows -> SMEM -> Write in rows\n");
    CHECK(cudaMemset(d_output, 0, inputMemSize));

    size_t sharedMemSize = blockDim2D.x * blockDim2D.y * sizeof(int);
    timer.Start();
    transposeCoalesced<<<gridDim2D, blockDim2D, sharedMemSize>>>(d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    #if DEBUG
    printf("Output matrix:\n");
    printArray(deviceOutput, numRows, numCols);
    printf("\n\n");
    #else
    printf("\n");
    #endif
    */
    // <<<< ==================================================

    // >>>> ==================================================
    /*
    printf("Transpose by Device: Read in rows -> SMEM with Padding -> Write in rows\n");
    CHECK(cudaMemset(d_output, 0, inputMemSize));

    sharedMemSize = blockDim2D.y * (blockDim2D.x + PAD) * sizeof(int);
    timer.Start();
    transposeCoalescedConflictFree<<<gridDim2D, blockDim2D, sharedMemSize>>>(d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    printf("\n");
    */
    // <<<< ==================================================

    // >>>> ==================================================
    printf("Transpose by Device: Read in rows (Unroll 2) -> SMEM (Rectangle, Padded) -> Write in rows (Unroll 2)\n");
    CHECK(cudaMemset(d_output, 0, inputMemSize));

    sharedMemSize = blockDimRectangle.y * (blockDimRectangle.x * 2 + PAD) * sizeof(int);
    timer.Start();
    transposeCoalescedConflictFreeUnroll2<<<gridDimUnroll2, blockDimRectangle, sharedMemSize>>>(d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    printf("\n");
    // <<<< ==================================================

    // >>>> ==================================================
    printf("Transpose by Device: Read in rows (Unroll 4 -> SMEM (Rectangle, Padded) -> Write in rows (Unroll 4)\n");
    CHECK(cudaMemset(d_output, 0, inputMemSize));

    sharedMemSize = blockDimRectangle.y * (blockDimRectangle.x * 4 + PAD) * sizeof(int);
    timer.Start();
    transposeCoalescedConflictFreeUnroll4<<<gridDimUnroll4, blockDimRectangle, sharedMemSize>>>(d_input, numCols, numRows, d_output);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    CHECK(cudaMemcpy(deviceOutput, d_output, inputMemSize, cudaMemcpyDeviceToHost));
    checkCorrectness(hostOutput, deviceOutput, numElements);
    printf("\n");
    // <<<< ==================================================

    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_input));
    delete[] input;
    delete[] hostOutput;
    delete[] deviceOutput;

    return EXIT_SUCCESS;
}
