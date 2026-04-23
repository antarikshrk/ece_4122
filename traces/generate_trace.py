# Author: Shiva Subramanian, Antariksh Krishnan
# GTID: 903780288
# Class: ECE 4122
# Last Date Modified: 4/23/26
"""
Description:
Generates synthetic branch trace files for testing the predictor pipeline.

Each record:
- 64-bit program counter
- 1-byte outcome (taken/not taken)
- 7 bytes padding (ensures 16-byte alignment)

Optimized using buffered writes for high I/O performance.
"""

import struct
import random
import argparse
import sys


def generate_trace(num_records, output_file):
    print(f"Generating {num_records} branch records to {output_file}...")

    try:
        with open(output_file, 'wb') as f:
            chunk_size = 100000
            buffer = bytearray()

            for i in range(num_records):
                # Random 64-bit PC
                pc = random.getrandbits(64)

                # Bias toward taken branches (~70%)
                outcome = 1 if random.random() < 0.7 else 0

                # Pack into 16-byte format: little-endian
                # Q = uint64, B = uint8, 7x = padding
                buffer.extend(struct.pack('<QB7x', pc, outcome))

                # Flush buffer periodically to reduce I/O overhead
                if (i + 1) % chunk_size == 0:
                    f.write(buffer)
                    buffer.clear()

            # Write remaining data
            if buffer:
                f.write(buffer)

        print("Trace generation complete.")

    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic branch traces.")
    parser.add_argument('--count', type=int, default=1000000)
    parser.add_argument('--out', type=str, default="traces/big.trace")

    args = parser.parse_args()

    generate_trace(args.count, args.out)