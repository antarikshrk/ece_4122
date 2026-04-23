/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description: Advanced streaming pipeline. Synchronizes disk I/O directly with the CUDA execution bus.
*/

#include "transfer.cuh"
#include "predictors.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

#define CUDA_CHECK(call)                                                                                                    \
    do                                                                                                                      \
    {                                                                                                                       \
        cudaError_t _err = (call);                                                                                          \
        if (_err != cudaSuccess)                                                                                            \
        {                                                                                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(_err) << std::endl; \
            exit(EXIT_FAILURE);                                                                                             \
        }                                                                                                                   \
    } while (0)

void transferAndPredict(TraceReader &reader, int *d_bht, uint8_t *h_total_results, size_t numRecords, int tableSize, int chunkSize, int threadsPerBlock, int useAtomics)
{
    std::vector<int> initBHT(tableSize, 2);
    CUDA_CHECK(cudaMemcpy(d_bht, initBHT.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice));

    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];

    BranchRecord *h_records_chunks[NUM_STREAMS];
    BranchRecord *d_records_chunks[NUM_STREAMS];
    uint8_t *d_results_chunks[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMallocHost(&h_records_chunks[i], chunkSize * sizeof(BranchRecord)));
        CUDA_CHECK(cudaMalloc(&d_records_chunks[i], chunkSize * sizeof(BranchRecord)));
        CUDA_CHECK(cudaMalloc(&d_results_chunks[i], chunkSize * sizeof(uint8_t)));
    }

    reader.reset();
    size_t numChunks = (numRecords + chunkSize - 1) / chunkSize;

    for (size_t i = 0; i < numChunks; ++i)
    {
        int s = i % NUM_STREAMS;

        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

        size_t count = reader.readChunk(h_records_chunks[s], chunkSize);
        if (count == 0)
            break;

        size_t offset = i * chunkSize;

        CUDA_CHECK(cudaMemcpyAsync(d_records_chunks[s], h_records_chunks[s], count * sizeof(BranchRecord), cudaMemcpyHostToDevice, streams[s]));

        int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
        bimodalGPUKernel<<<blocks, threadsPerBlock, 0, streams[s]>>>(d_records_chunks[s], d_bht, d_results_chunks[s], count, tableSize, useAtomics);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_total_results + offset, d_results_chunks[s], count * sizeof(uint8_t), cudaMemcpyDeviceToHost, streams[s]));
    }

    // Explicitly block host to catch asynchronous kernel exceptions before teardown
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams[i])); // Ensures all pending ops in streams finish safely
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFreeHost(h_records_chunks[i]));
        CUDA_CHECK(cudaFree(d_records_chunks[i]));
        CUDA_CHECK(cudaFree(d_results_chunks[i]));
    }
}