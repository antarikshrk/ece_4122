/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26
*/

#pragma once
#include "trace_reader.h"
#include <cstdint>
#include <cstddef>

void bimodalCPU(TraceReader &reader, size_t numRecords, int tableSize, int *bht, uint8_t *results);