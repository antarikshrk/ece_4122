/*
Description:
CUDA kernel implementing the bimodal branch predictor.

Each thread:
1. Reads one branch record
2. Computes prediction using BHT
3. Writes correctness result
4. Updates BHT (atomic or non-atomic)

Tradeoffs:
- Atomic mode: correct but slower (serialization)
- Non-atomic: faster but introduces race conditions
*/

#include "predictors.cuh"

__global__ void bimodalGPUKernel(const BranchRecord *d_records,
                                 int *d_bht,
                                 uint8_t *d_results,
                                 size_t numRecords,
                                 int tableSize,
                                 int useAtomics)
{
    // Compute global thread index
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRecords)
        return;

    // Index into BHT using lower PC bits
    int index = d_records[tid].pc & (tableSize - 1);

    // Predict taken if counter >= 2
    bool prediction = (d_bht[index] >= 2);

    // Store prediction correctness
    d_results[tid] = (prediction == d_records[tid].outcome) ? 1 : 0;

    // -------- BHT UPDATE --------
    if (useAtomics == 1)
    {
        // Atomic CAS loop ensures correctness under concurrent updates
        int oldVal, assumed;

        if (d_records[tid].outcome == 1)
        {
            oldVal = d_bht[index];
            do
            {
                assumed = oldVal;
                int newVal = assumed + (assumed < 3);
                oldVal = atomicCAS(&d_bht[index], assumed, newVal);
            } while (assumed != oldVal);
        }
        else
        {
            oldVal = d_bht[index];
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
        // Non-atomic (faster, but unsafe under contention)
        if (d_records[tid].outcome == 1)
            d_bht[index] += (d_bht[index] < 3);
        else
            d_bht[index] -= (d_bht[index] > 0);
    }
}