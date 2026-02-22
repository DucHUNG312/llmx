#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <sstream>

#ifndef NDEBUG
#define LLMX_DEBUG
#else
#endif

namespace llmx {

class Logger {
public:
  enum class LogSeverity : int8_t {
    kFATAL = 0,
    kERROR,
    kWARNING,
    kINFO,
    kDEBUG,
    kVERBOSE
  };

  Logger(LogSeverity severity = LogSeverity::kDEBUG, bool color = true);

  void log(LogSeverity severity, const std::string &msg,
           std::filesystem::path path = "") noexcept;

  [[nodiscard]] LogSeverity get_reportable_severity() const noexcept;

  void set_reportable_log_severity(LogSeverity severity) noexcept;

  void set_log_color(bool color) noexcept;

private:
  LogSeverity severity_;
  bool color_;
};

Logger &get_logger() noexcept;

#define LLMX_COLOR_NORMAL "\033[0m";
#define LLMX_COLOR_RED "\033[0;31m";
#define LLMX_COLOR_YELLOW "\033[0;33m";
#define LLMX_COLOR_GREEN "\033[0;32m";
#define LLMX_COLOR_MAGENTA "\033[1;35m";

} // namespace llmx

#ifdef LLMX_DEBUG
#define LLMX_LOG(logger, severity, msg)                                        \
  do {                                                                         \
    std::stringstream sstream{};                                               \
    sstream << msg;                                                            \
    logger.log(severity, sstream.str());                                       \
  } while (0)
#else
#define LLMX_LOG(logger, severity, msg) ((void)0)
#endif

#define LLMX_LOG_VERBOSE(msg)                                                  \
  LLMX_LOG(::llmx::get_logger(), ::llmx::Logger::LogSeverity::kVERBOSE, msg)
#define LLMX_LOG_DEBUG(msg)                                                    \
  LLMX_LOG(::llmx::get_logger(), ::llmx::Logger::LogSeverity::kDEBUG, msg)
#define LLMX_LOG_INFO(msg)                                                     \
  LLMX_LOG(::llmx::get_logger(), ::llmx::Logger::LogSeverity::kINFO, msg)
#define LLMX_LOG_WARN(msg)                                                     \
  LLMX_LOG(::llmx::get_logger(), ::llmx::Logger::LogSeverity::kWARNING, msg)

#ifdef LLMX_DEBUG
#define LLMX_LOG_ERROR(msg)                                                    \
  LLMX_LOG(::llmx::get_logger(), ::llmx::Logger::LogSeverity::kERROR, msg)
#define LLMX_LOG_FATAL(msg)                                                    \
  do {                                                                         \
    LLMX_LOG_ERROR(msg);                                                       \
    throw;                                                                     \
  } while (0)
#else
#define LLMX_LOG_ERROR(msg)                                                    \
  do {                                                                         \
    std::stringstream sstream{};                                               \
    sstream << msg;                                                            \
    ::llmx::get_logger().log(::llmx::Logger::LogSeverity::kERROR,              \
                             sstream.str());                                   \
  } while (0)
#define LLMX_LOG_FATAL(msg)                                                    \
  do {                                                                         \
    LLMX_LOG_ERROR(msg);                                                       \
    throw;                                                                     \
  } while (0)
#endif

#ifdef LLMX_DEBUG
#define LLMX_CHECK_GET_MACRO(_1, _2, NAME, ...) NAME
#define LLMX_CHECK_1(cond)                                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      LLMX_LOG_FATAL("Check " << #cond << " failed in " << __FILE__ << "("     \
                              << __LINE__ << ")");                             \
    }                                                                          \
  } while (0)
#define LLMX_CHECK_2(cond, msg)                                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      LLMX_LOG_FATAL("Check " << #cond << " failed: " << msg << " in "         \
                              << __FILE__ << "(" << __LINE__ << ")");          \
    }                                                                          \
  } while (0)
#define LLMX_CHECK(...)                                                        \
  LLMX_CHECK_GET_MACRO(__VA_ARGS__, LLMX_CHECK_2, LLMX_CHECK_1)(__VA_ARGS__)
#define LLMX_CUDA_CHECK(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    LLMX_CHECK(err == cudaSuccess, cudaGetErrorString(err));                   \
  } while (0)
#define LLMX_CUDA_CHECK_LAST_ERROR()                                           \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    LLMX_CHECK(err == cudaSuccess, cudaGetErrorString(err));                   \
  } while (0)
#define LLMX_CUDA_CHECK_SYNC(call)                                             \
  do {                                                                         \
    LLMX_CUDA_CHECK(call);                                                     \
    LLMX_CUDA_CHECK(cudaDeviceSynchronize());                                  \
  } while (0)
#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      LLMX_LOG_FATAL("cuBLAS error in " << __FILE__ << "(" << __LINE__         \
                                        << "): "                               \
                                        << "Error code " << status);           \
    }                                                                          \
  } while (0)
#else
#define LLMX_CHECK(...) ((void)0)
#define LLMX_CUDA_CHECK(call) call
#define LLMX_CUDA_CHECK_LAST_ERROR() ((void)0)
#define LLMX_CUDA_CHECK_SYNC(call) call
#define CUBLAS_CHECK(call) call

#endif

inline int cuda_device_count() {
  int device_count = 0;
  LLMX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  return device_count;
}