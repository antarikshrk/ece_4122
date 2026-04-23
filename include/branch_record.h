/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description: Defines the shared BranchRecord structure used across CPU and GPU modules. Uses bitwise math for Endianness guards.
*/

#pragma once
#include <cstdint>

// alignas(16) strictly enforces the 16-byte boundary for GPU cache lines.
struct alignas(16) BranchRecord
{
    uint64_t pc;        // Program counter (branch address)
    uint8_t outcome;    // 1 = branch taken, 0 = not taken
    uint8_t padding[7]; // Pads struct to exactly 16 bytes

    inline void fixEndianness()
    {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        pc = ((pc & 0x00000000000000FFULL) << 56) |
             ((pc & 0x000000000000FF00ULL) << 40) |
             ((pc & 0x0000000000FF0000ULL) << 24) |
             ((pc & 0x00000000FF000000ULL) << 8) |
             ((pc & 0x000000FF00000000ULL) >> 8) |
             ((pc & 0x0000FF0000000000ULL) >> 24) |
             ((pc & 0x00FF000000000000ULL) >> 40) |
             ((pc & 0xFF00000000000000ULL) >> 56);
#endif
    }
};