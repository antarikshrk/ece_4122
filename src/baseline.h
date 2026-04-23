/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description:
Declares the CPU baseline implementation of the bimodal branch predictor.
*/

#pragma once
#include "trace_reader.h"
#include <cstdint>
#include <cstddef>

/*
Runs the CPU-based bimodal predictor for validation and benchmarking.

See baseline.cpp for full behavior details.
*/
void bimodalCPU(TraceReader &reader, size_t numRecords, int tableSize, int *bht, uint8_t *results);