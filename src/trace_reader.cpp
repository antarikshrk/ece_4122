/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description: Stateful disk streaming class. Feeds data directly to the GPU pipeline.
*/

#include "trace_reader.h"
#include <iostream>
#include <cstdlib>

TraceReader::TraceReader(const char *fname) : filename(fname), totalRecords(0)
{
    file.open(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open trace file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::streamsize fileSize = file.tellg();
    if (fileSize <= 0 || fileSize % sizeof(BranchRecord) != 0)
    {
        std::cerr << "Error: Trace file is empty or corrupted (size mismatch)." << std::endl;
        exit(EXIT_FAILURE);
    }

    totalRecords = fileSize / sizeof(BranchRecord);
    file.seekg(0, std::ios::beg);
}

TraceReader::~TraceReader()
{
    if (file.is_open())
    {
        file.close();
    }
}

size_t TraceReader::getTotalRecords() const
{
    return totalRecords;
}

void TraceReader::reset()
{
    file.clear();
    file.seekg(0, std::ios::beg);
}

size_t TraceReader::readChunk(BranchRecord *buffer, size_t maxRecords)
{
    if (!file.good())
        return 0;

    file.read(reinterpret_cast<char *>(buffer), maxRecords * sizeof(BranchRecord));
    size_t bytesRead = file.gcount();
    size_t recordsRead = bytesRead / sizeof(BranchRecord);

    for (size_t i = 0; i < recordsRead; ++i)
    {
        buffer[i].fixEndianness();
    }

    return recordsRead;
}