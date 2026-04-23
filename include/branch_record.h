/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description:
Defines a compact, GPU-friendly data structure representing a single branch event.
The struct is explicitly aligned and padded to ensure efficient memory access on
both CPU and GPU architectures. Includes a utility to correct endianness when
running on big-endian systems.
*/

#pragma once
#include <cstdint>

// Enforce 16-byte alignment for optimal GPU memory transactions and cache usage.
struct alignas(16) BranchRecord
{
    uint64_t pc;        // Program counter (address of the branch instruction)
    uint8_t outcome;    // Branch outcome: 1 = taken, 0 = not taken
    uint8_t padding[7]; // Padding to ensure total struct size is exactly 16 bytes

    /*
    Function: fixEndianness

    Description:
    Converts the program counter (pc) from big-endian to little-endian format
    if the system architecture is big-endian. This ensures consistent data
    interpretation across heterogeneous systems (e.g., CPU ↔ GPU transfers).

    Notes:
    - No operation is performed on little-endian systems.
    - Uses bitwise masking and shifting to reverse byte order.
    */
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