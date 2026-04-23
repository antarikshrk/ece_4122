/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26
*/

#include "validator.h"
#include <iostream>
#include <iomanip>
#include <cmath>

void validateResults(const uint8_t *gpuResults, const uint8_t *cpuResults, size_t numRecords, int useAtomics)
{
    size_t mismatches = 0;
    size_t gpuCorrect = 0;
    size_t cpuCorrect = 0;

    for (size_t i = 0; i < numRecords; ++i)
    {
        if (gpuResults[i] != cpuResults[i])
            mismatches++;
        if (gpuResults[i] == 1)
            gpuCorrect++;
        if (cpuResults[i] == 1)
            cpuCorrect++;
    }

    double gpuAccuracy = static_cast<double>(gpuCorrect) / numRecords * 100.0;
    double cpuAccuracy = static_cast<double>(cpuCorrect) / numRecords * 100.0;
    double delta = std::abs(gpuAccuracy - cpuAccuracy);

    std::cout << "======================================" << std::endl;
    std::cout << "VALIDATION RESULTS" << std::endl;
    std::cout << "Mode:            " << (useAtomics ? "ATOMIC (Correctness)" : "NO ATOMIC (Speed)") << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU Accuracy:    " << gpuAccuracy << "%" << std::endl;
    std::cout << "CPU Accuracy:    " << cpuAccuracy << "%" << std::endl;
    std::cout << "Delta:           " << delta << "%" << std::endl;
    std::cout << "Mismatches:      " << mismatches << " out of " << numRecords << std::endl;
    std::cout << "======================================" << std::endl;
}