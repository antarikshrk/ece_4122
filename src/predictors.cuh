/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26
*/

#pragma once
#include "branch_record.h"

__global__ void bimodalGPUKernel(const BranchRecord *d_records, int *d_bht, uint8_t *d_results, size_t numRecords, int tableSize, int useAtomics);