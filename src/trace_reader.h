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

/**
 * @class TraceReader
 * @brief Manages stateful binary streaming of branch trace records from disk.
 * * This class provides an interface to read branch records in manageable chunks,
 * abstracting the file I/O and providing endianness correction to ensure 
 * data integrity across different hardware architectures.
 */
class TraceReader
{
private:
    std::ifstream file;      ///< Binary input stream for the trace file.
    size_t totalRecords;     ///< Cached total number of records in the file.
    const char *filename;    ///< Path to the source trace file.

public:
    /**
     * @brief Constructs a reader and validates the trace file.
     * @param filename Path to the .trace binary file.
     * @note Exits the program if the file cannot be opened or is corrupted.
     */
    TraceReader(const char *filename);

    /** @brief Destructor to ensure the file handle is properly closed. */
    ~TraceReader();

    /** @return Total number of BranchRecord entries found in the file. */
    size_t getTotalRecords() const;

    /** @brief Seeks back to the beginning of the file to allow multi-pass simulation. */
    void reset();

    /**
     * @brief Reads a specific number of records into a provided buffer.
     * @param buffer Pointer to the destination memory (usually host pinned memory).
     * @param maxRecords Maximum number of BranchRecord objects to read.
     * @return The actual number of records successfully read and processed.
     */
    size_t readChunk(BranchRecord *buffer, size_t maxRecords);
};