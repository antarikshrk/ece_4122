/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26
*/

#pragma once
#include "branch_record.h"
#include <fstream>
#include <cstddef>

class TraceReader
{
private:
    std::ifstream file;
    size_t totalRecords;
    const char *filename;

public:
    TraceReader(const char *filename);
    ~TraceReader();

    size_t getTotalRecords() const;
    void reset();
    size_t readChunk(BranchRecord *buffer, size_t maxRecords);
};