/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26
*/

#include "predictors.cuh"

__global__ void bimodalGPUKernel(const BranchRecord *d_records, int *d_bht, uint8_t *d_results, size_t numRecords, int tableSize, int useAtomics)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRecords)
        return;

    int index = d_records[tid].pc & (tableSize - 1);
    bool prediction = (d_bht[index] >= 2);

    d_results[tid] = (prediction == d_records[tid].outcome) ? 1 : 0;

    if (useAtomics == 1)
    {
        if (d_records[tid].outcome == 1)
        {
            int oldVal = d_bht[index];
            int assumed;
            do
            {
                assumed = oldVal;
                int newVal = assumed + (assumed < 3);
                oldVal = atomicCAS(&d_bht[index], assumed, newVal);
            } while (assumed != oldVal);
        }
        else
        {
            int oldVal = d_bht[index];
            int assumed;
            do
            {
                assumed = oldVal;
                int newVal = assumed - (assumed > 0);
                oldVal = atomicCAS(&d_bht[index], assumed, newVal);
            } while (assumed != oldVal);
        }
    }
    else
    {
        if (d_records[tid].outcome == 1)
            d_bht[index] += (d_bht[index] < 3);
        else
            d_bht[index] -= (d_bht[index] > 0);
    }
}