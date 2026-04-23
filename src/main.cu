/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description: Entry point. Manages CLI parsing, PCIe throttle limits, and pipeline execution.
*/

#include "trace_reader.h"
#include "baseline.h"
#include "transfer.cuh"
#include "validator.h"
#include "metrics.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

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

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        std::cerr << "Usage: ./predictor <trace_file> <table_size> <threads_per_block> <chunk_size> <use_atomics (0|1)>" << std::endl;
        return 1;
    }

    const char *traceFile = argv[1];
    int tableSize = 0, threadsPerBlock = 0, chunkSize = 0, useAtomics = 0;

    try
    {
        tableSize = std::stoi(argv[2]);
        threadsPerBlock = std::stoi(argv[3]);
        chunkSize = std::stoi(argv[4]);
        useAtomics = std::stoi(argv[5]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: Invalid numeric argument provided." << std::endl;
        return 1;
    }

    if (tableSize <= 0 || (tableSize & (tableSize - 1)) != 0)
    {
        std::cerr << "Error: Table size must be a positive power of 2 (e.g., 1024, 2048)." << std::endl;
        return 1;
    }
    if (threadsPerBlock <= 0 || threadsPerBlock > 1024)
    {
        std::cerr << "Error: Threads per block must be between 1 and 1024." << std::endl;
        return 1;
    }
    if (chunkSize > 10000000)
    {
        std::cerr << "Error: Chunk size exceeds VRAM safety limits (Max 10,000,000)." << std::endl;
        return 1;
    }
    if (chunkSize < 32768)
    {
        std::cout << "[WARNING] Chunk size (" << chunkSize << ") is too small and will bottleneck the PCIe bus. Auto-scaling to 32,768." << std::endl;
        chunkSize = 32768;
    }

    std::cout << "Initializing Branch Prediction Simulator..." << std::endl;

    TraceReader reader(traceFile);
    size_t numRecords = reader.getTotalRecords();
    std::cout << "Loaded stream with " << numRecords << " records." << std::endl;

    uint8_t *h_gpuResults;
    CUDA_CHECK(cudaMallocHost(&h_gpuResults, numRecords * sizeof(uint8_t)));
    std::memset(h_gpuResults, 0, numRecords * sizeof(uint8_t));

    std::vector<uint8_t> h_cpuResults(numRecords, 0);
    std::vector<int> h_cpuBHT(tableSize, 2);

    int *d_bht;
    CUDA_CHECK(cudaMalloc(&d_bht, tableSize * sizeof(int)));

    // GPU run
    std::cout << "\nRunning GPU Pipeline..." << std::endl;
    auto gpuStart = startTimer();
    transferAndPredict(reader, d_bht, h_gpuResults, numRecords, tableSize, chunkSize, threadsPerBlock, useAtomics);
    double gpuTime = stopTimerMs(gpuStart);
    std::cout << "GPU Time: " << gpuTime << " ms" << std::endl;

    // CPU run
    std::cout << "\nRunning CPU Baseline..." << std::endl;
    auto cpuStart = startTimer();
    bimodalCPU(reader, numRecords, tableSize, h_cpuBHT.data(), h_cpuResults.data());
    double cpuTime = stopTimerMs(cpuStart);
    std::cout << "CPU Time: " << cpuTime << " ms" << std::endl;

    // Metrics
    double speedup = calcSpeedup(cpuTime, gpuTime);
    std::cout << "\nCalculated Speedup: " << speedup << "x" << std::endl;

    validateResults(h_gpuResults, h_cpuResults.data(), numRecords, useAtomics);

    CUDA_CHECK(cudaFree(d_bht));
    CUDA_CHECK(cudaFreeHost(h_gpuResults));

    return 0;
}