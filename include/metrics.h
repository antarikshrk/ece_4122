/*
Author: Shiva Subramanian, Antariksh Krishnan
GTID: 903780288
Class: ECE 4122
Last Date Modified: 4/23/26
*/

#pragma once
#include <chrono>

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline TimePoint startTimer()
{
    return Clock::now();
}

inline double stopTimerMs(const TimePoint &start)
{
    auto end = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

inline double calcSpeedup(double serialTime, double parallelTime)
{
    if (parallelTime <= 0.0)
        return 0.0;
    return serialTime / parallelTime;
}