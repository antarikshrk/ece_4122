/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description:
Main driver for the branch prediction simulator.

Responsibilities:
- Parses and validates command-line arguments
- Allocates host and device memory
- Executes GPU pipeline (streaming + kernel)
- Executes CPU baseline for comparison
- Measures runtime and computes speedup
- Validates correctness between CPU and GPU outputs

Design Notes:
- Uses pinned (page-locked) memory for faster PCIe transfers
- Enforces constraints on chunk size to avoid VRAM overflow
- Supports atomic and non-atomic GPU execution modes
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

// Macro for CUDA error handling (fail-fast on any runtime error)
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
    // Validate CLI arguments
    if (argc < 6)
    {
        std::cerr << "Usage: ./predictor <trace_file> <table_size> <threads_per_block> <chunk_size> <use_atomics (0|1)>" << std::endl;
        return 1;
    }

    const char *traceFile = argv[1];
    int tableSize = 0, threadsPerBlock = 0, chunkSize = 0, useAtomics = 0;

    // Parse numeric arguments safely
    try
    {
        tableSize = std::stoi(argv[2]);
        threadsPerBlock = std::stoi(argv[3]);
        chunkSize = std::stoi(argv[4]);
        useAtomics = std::stoi(argv[5]);
    }
    catch (...)
    {
        std::cerr << "Error: Invalid numeric argument provided." << std::endl;
        return 1;
    }

    // Validate constraints
    if (tableSize <= 0 || (tableSize & (tableSize - 1)) != 0)
    {
        std::cerr << "Error: Table size must be a power of 2." << std::endl;
        return 1;
    }

    if (threadsPerBlock <= 0 || threadsPerBlock > 1024)
    {
        std::cerr << "Error: Threads per block must be between 1 and 1024." << std::endl;
        return 1;
    }

    // Prevent excessive VRAM usage
    if (chunkSize > 10000000)
    {
        std::cerr << "Error: Chunk size exceeds safe VRAM limits." << std::endl;
        return 1;
    }

    // Avoid PCIe underutilization
    if (chunkSize < 32768)
    {
        std::cout << "[WARNING] Chunk size too small. Auto-scaling to 32,768." << std::endl;
        chunkSize = 32768;
    }

    std::cout << "Initializing Branch Prediction Simulator..." << std::endl;

    // Initialize trace reader
    TraceReader reader(traceFile);
    size_t numRecords = reader.getTotalRecords();

    std::cout << "Loaded stream with " << numRecords << " records." << std::endl;

    // Allocate pinned host memory for GPU results (faster transfers)
    uint8_t *h_gpuResults;
    CUDA_CHECK(cudaMallocHost(&h_gpuResults, numRecords * sizeof(uint8_t)));
    std::memset(h_gpuResults, 0, numRecords * sizeof(uint8_t));

    // CPU-side data structures
    std::vector<uint8_t> h_cpuResults(numRecords, 0);
    std::vector<int> h_cpuBHT(tableSize, 2);

    // Allocate GPU BHT
    int *d_bht;
    CUDA_CHECK(cudaMalloc(&d_bht, tableSize * sizeof(int)));

    // ---------------- GPU EXECUTION ----------------
    std::cout << "\nRunning GPU Pipeline..." << std::endl;

    auto gpuStart = startTimer();

    // Runs streaming pipeline: disk → CPU → GPU → CPU
    transferAndPredict(reader, d_bht, h_gpuResults,
                       numRecords, tableSize,
                       chunkSize, threadsPerBlock, useAtomics);

    double gpuTime = stopTimerMs(gpuStart);
    std::cout << "GPU Time: " << gpuTime << " ms" << std::endl;

    // ---------------- CPU EXECUTION ----------------
    std::cout << "\nRunning CPU Baseline..." << std::endl;

    auto cpuStart = startTimer();

    bimodalCPU(reader, numRecords, tableSize,
               h_cpuBHT.data(), h_cpuResults.data());

    double cpuTime = stopTimerMs(cpuStart);
    std::cout << "CPU Time: " << cpuTime << " ms" << std::endl;

    // ---------------- METRICS ----------------
    double speedup = calcSpeedup(cpuTime, gpuTime);
    std::cout << "\nCalculated Speedup: " << speedup << "x" << std::endl;

    // ---------------- VALIDATION ----------------
    validateResults(h_gpuResults, h_cpuResults.data(), numRecords, useAtomics);

    // Cleanup resources
    CUDA_CHECK(cudaFree(d_bht));
    CUDA_CHECK(cudaFreeHost(h_gpuResults));

    return 0;
}