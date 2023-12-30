#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << msg << std::endl;
    }
};

extern Logger gLogger;

#endif // LOGGER_HPP

