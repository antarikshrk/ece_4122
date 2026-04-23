/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26
*/

#pragma once
#include "trace_reader.h"
#include <cstddef>
#include <cstdint>

void transferAndPredict(TraceReader &reader, int *d_bht, uint8_t *h_total_results, size_t numRecords, int tableSize, int chunkSize, int threadsPerBlock, int useAtomics);