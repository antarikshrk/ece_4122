/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description:
Implements a serial CPU baseline for bimodal branch prediction.
Processes large traces in fixed-size chunks to avoid excessive memory usage
while maintaining correctness for comparison with the GPU pipeline.
*/

#include "baseline.h"
#include <vector>

/*
Function: bimodalCPU

Description:
Executes a bimodal branch predictor on the CPU using a 2-bit saturating counter table.
The trace is streamed in chunks to reduce memory footprint.

Parameters:
- reader: TraceReader instance for streaming branch records
- numRecords: Total number of records (used for indexing results)
- tableSize: Size of the Branch History Table (must be power of 2)
- bht: Pointer to branch history table (2-bit saturating counters)
- results: Output array storing prediction correctness (1 = correct, 0 = incorrect)

Notes:
- Uses lower bits of PC for indexing (masking via tableSize - 1)
- Saturating counters range from [0,3]
- Chunking avoids loading entire trace into RAM
*/
void bimodalCPU(TraceReader &reader, size_t numRecords, int tableSize, int *bht, uint8_t *results)
{
    // Initialize BHT to weakly taken (2)
    for (int i = 0; i < tableSize; ++i)
    {
        bht[i] = 2;
    }

    reader.reset();

    // Chunk size chosen to balance cache locality and memory usage
    const size_t CPU_CHUNK_SIZE = 1000000;
    std::vector<BranchRecord> buffer(CPU_CHUNK_SIZE);

    size_t offset = 0;

    // Stream through trace file
    while (true)
    {
        size_t count = reader.readChunk(buffer.data(), CPU_CHUNK_SIZE);
        if (count == 0)
            break;

        for (size_t i = 0; i < count; ++i)
        {
            // Index into BHT using lower bits of PC
            int index = buffer[i].pc & (tableSize - 1);

            // Predict taken if counter >= 2
            bool prediction = (bht[index] >= 2);

            // Store correctness
            results[offset + i] = (prediction == buffer[i].outcome) ? 1 : 0;

            // Update saturating counter
            if (buffer[i].outcome == 1)
                bht[index] += (bht[index] < 3);
            else
                bht[index] -= (bht[index] > 0);
        }

        offset += count;
    }
}