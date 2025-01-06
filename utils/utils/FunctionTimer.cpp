#include "FunctionTimer.h"

std::unordered_map<std::string, double> FunctionTimer::timings;
std::mutex FunctionTimer::timingsMutex;

FunctionTimer::FunctionTimer(const std::string& functionName)
    : functionName(functionName),
      startTime(std::chrono::high_resolution_clock::now()) {}

FunctionTimer::~FunctionTimer() {
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    std::lock_guard<std::mutex> lock(timingsMutex);
    timings[functionName] += duration / 1000.0; // Convert to milliseconds
}

void FunctionTimer::PrintTimings() {
    std::lock_guard<std::mutex> lock(timingsMutex);
    std::cout << "Function Execution Times:" << std::endl;
    for (const auto& [name, time] : timings) {
        std::cout << name << ": " << time << " ms" << std::endl;
    }
}
