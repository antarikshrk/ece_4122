/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26
*/

#pragma once
#include <cstddef>
#include <cstdint>

void validateResults(const uint8_t *gpuResults, const uint8_t *cpuResults, size_t numRecords, int useAtomics);