/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26

Description:
Provides lightweight timing and performance measurement utilities.
Used to benchmark execution time and compute speedup between
serial and parallel implementations.
*/

#pragma once
#include <chrono>

// Type aliases for improved readability when working with time points.
using Clock = std::chrono::steady_clock;              // Monotonic clock (not affected by system time changes)
using TimePoint = std::chrono::time_point<Clock>;     // Represents a specific point in time

/*
Function: startTimer

Description:
Captures the current time point to mark the beginning of a timed section.

Returns:
- A TimePoint representing the current time.
*/
inline TimePoint startTimer()
{
    return Clock::now();
}

/*
Function: stopTimerMs

Description:
Computes the elapsed time in milliseconds since the provided start time.

Parameters:
- start: TimePoint marking the beginning of the measurement

Returns:
- Elapsed time in milliseconds as a double
*/
inline double stopTimerMs(const TimePoint &start)
{
    auto end = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

/*
Function: calcSpeedup

Description:
Calculates performance speedup achieved by a parallel implementation
relative to a serial baseline.

Formula:
    speedup = serialTime / parallelTime

Parameters:
- serialTime: Execution time of the serial version (ms)
- parallelTime: Execution time of the parallel version (ms)

Returns:
- Speedup factor (greater than 1 indicates improvement)
- Returns 0.0 if parallelTime is invalid (<= 0) to avoid division by zero
*/
inline double calcSpeedup(double serialTime, double parallelTime)
{
    if (parallelTime <= 0.0)
        return 0.0;
    return serialTime / parallelTime;
}