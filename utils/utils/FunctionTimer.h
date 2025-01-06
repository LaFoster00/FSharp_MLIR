//
// Created by lasse on 26/12/2024.
//

#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <mutex>

class FunctionTimer {
public:
    FunctionTimer(const std::string& functionName);
    ~FunctionTimer();

    static void PrintTimings();

private:
    static std::unordered_map<std::string, double> timings;
    static std::mutex timingsMutex;

    std::string functionName;
    std::chrono::high_resolution_clock::time_point startTime;
};
