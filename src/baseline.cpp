/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description: Serial CPU implementation updated to support chunked streaming.
*/

#include "baseline.h"
#include <vector>

void bimodalCPU(TraceReader &reader, size_t numRecords, int tableSize, int *bht, uint8_t *results)
{
    for (int i = 0; i < tableSize; ++i)
    {
        bht[i] = 2;
    }

    reader.reset();

    // Process CPU baseline in 1 million record chunks to protect RAM
    const size_t CPU_CHUNK_SIZE = 1000000;
    std::vector<BranchRecord> buffer(CPU_CHUNK_SIZE);

    size_t offset = 0;
    while (true)
    {
        size_t count = reader.readChunk(buffer.data(), CPU_CHUNK_SIZE);
        if (count == 0)
            break;

        for (size_t i = 0; i < count; ++i)
        {
            int index = buffer[i].pc & (tableSize - 1);
            bool prediction = (bht[index] >= 2);

            results[offset + i] = (prediction == buffer[i].outcome) ? 1 : 0;

            if (buffer[i].outcome == 1)
                bht[index] += (bht[index] < 3);
            else
                bht[index] -= (bht[index] > 0);
        }
        offset += count;
    }
}