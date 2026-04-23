/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description:
Implements a multi-stream CUDA pipeline that overlaps disk I/O, memory transfers,
and kernel execution to maximize throughput.

Pipeline Stages (per stream):
1. Read chunk from disk into pinned host memory
2. Asynchronously copy host → device (H2D)
3. Launch GPU kernel for prediction
4. Asynchronously copy results device → host (D2H)

Key Design Choices:
- Uses multiple CUDA streams to enable concurrency
- Uses pinned (page-locked) host memory for faster PCIe transfers
- Reuses buffers in a round-robin fashion to avoid reallocations
- Synchronizes per-stream before reuse to prevent data hazards
*/

#include "transfer.cuh"
#include "predictors.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

// Macro for consistent CUDA error handling (fail-fast)
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

/*
Function: transferAndPredict

Description:
Coordinates the full GPU execution pipeline by streaming trace data in chunks,
processing it on the GPU, and collecting results asynchronously.

Parameters:
- reader: TraceReader for streaming branch records from disk
- d_bht: Device pointer to Branch History Table (shared across all chunks)
- h_total_results: Host buffer to store final prediction correctness results
- numRecords: Total number of branch records in trace
- tableSize: Size of BHT (must be power of 2)
- chunkSize: Number of records processed per iteration
- threadsPerBlock: CUDA block size for kernel launch
- useAtomics: Enables atomic updates (correctness vs performance tradeoff)

Notes:
- Stream count is fixed (NUM_STREAMS = 4) for balanced concurrency
- Chunking prevents VRAM exhaustion and improves pipeline efficiency
*/
void transferAndPredict(TraceReader &reader, int *d_bht, uint8_t *h_total_results, size_t numRecords, int tableSize, int chunkSize, int threadsPerBlock, int useAtomics)
{
    // Initialize BHT on device (all entries set to weakly taken = 2)
    std::vector<int> initBHT(tableSize, 2);
    CUDA_CHECK(cudaMemcpy(d_bht, initBHT.data(), tableSize * sizeof(int), cudaMemcpyHostToDevice));

    // Number of concurrent CUDA streams used for pipelining
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];

    // Per-stream buffers (host and device)
    BranchRecord *h_records_chunks[NUM_STREAMS]; // pinned host buffers
    BranchRecord *d_records_chunks[NUM_STREAMS]; // device buffers for input records
    uint8_t *d_results_chunks[NUM_STREAMS];      // device buffers for output results

    // Allocate resources for each stream
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

        // Pinned host memory enables faster DMA transfers over PCIe
        CUDA_CHECK(cudaMallocHost(&h_records_chunks[i], chunkSize * sizeof(BranchRecord)));

        // Device buffers for records and results
        CUDA_CHECK(cudaMalloc(&d_records_chunks[i], chunkSize * sizeof(BranchRecord)));
        CUDA_CHECK(cudaMalloc(&d_results_chunks[i], chunkSize * sizeof(uint8_t)));
    }

    // Reset reader to beginning of trace
    reader.reset();

    // Total number of chunks required to process entire trace
    size_t numChunks = (numRecords + chunkSize - 1) / chunkSize;

    // Main pipeline loop
    for (size_t i = 0; i < numChunks; ++i)
    {
        // Select stream in round-robin fashion
        int s = i % NUM_STREAMS;

        // Ensure previous operations in this stream are complete before reuse
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

        // Read next chunk from disk into host buffer
        size_t count = reader.readChunk(h_records_chunks[s], chunkSize);
        if (count == 0)
            break;

        // Compute global offset into output array
        size_t offset = i * chunkSize;

        // Asynchronous Host → Device transfer
        CUDA_CHECK(cudaMemcpyAsync(d_records_chunks[s],
                                   h_records_chunks[s],
                                   count * sizeof(BranchRecord),
                                   cudaMemcpyHostToDevice,
                                   streams[s]));

        // Configure kernel launch
        int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel on the same stream (ensures correct ordering)
        bimodalGPUKernel<<<blocks, threadsPerBlock, 0, streams[s]>>>(
            d_records_chunks[s],
            d_bht,
            d_results_chunks[s],
            count,
            tableSize,
            useAtomics);

        // Check for launch errors immediately
        CUDA_CHECK(cudaGetLastError());

        // Asynchronous Device → Host transfer of results
        CUDA_CHECK(cudaMemcpyAsync(h_total_results + offset,
                                   d_results_chunks[s],
                                   count * sizeof(uint8_t),
                                   cudaMemcpyDeviceToHost,
                                   streams[s]));
    }

    // Global synchronization:
    // Ensures all kernels and memory operations are completed
    // before cleanup and also surfaces any asynchronous errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup all allocated resources
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        // Ensure stream has finished all pending work
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        // Destroy stream and free associated memory
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFreeHost(h_records_chunks[i]));
        CUDA_CHECK(cudaFree(d_records_chunks[i]));
        CUDA_CHECK(cudaFree(d_results_chunks[i]));
    }
}