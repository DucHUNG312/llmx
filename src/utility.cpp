#include "llmx/utility.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace llmx {

Logger::Logger(LogSeverity severity, bool color)
    : severity_(severity), color_(color) {}

void Logger::log(LogSeverity severity, const std::string &msg,
                 std::filesystem::path path) noexcept {
  if (severity > severity_) {
    return;
  }

  if (color_) {
    switch (severity) {
    case LogSeverity::kFATAL:
    case LogSeverity::kERROR:
      std::cerr << LLMX_COLOR_RED;
      break;
    case LogSeverity::kWARNING:
      std::cerr << LLMX_COLOR_YELLOW;
      break;
    case LogSeverity::kINFO:
      std::cerr << LLMX_COLOR_GREEN;
      break;
    case LogSeverity::kDEBUG:
      std::cerr << LLMX_COLOR_MAGENTA;
      break;
    case LogSeverity::kVERBOSE:
      std::cerr << LLMX_COLOR_NORMAL;
      break;
    default:
      break;
    }
  }

  auto severity_to_string = [](LogSeverity s) {
    switch (s) {
    case LogSeverity::kFATAL:
      return "[FATAL] ";
    case LogSeverity::kERROR:
      return "[ERROR] ";
    case LogSeverity::kWARNING:
      return "[WARNING] ";
    case LogSeverity::kINFO:
      return "[INFO] ";
    case LogSeverity::kDEBUG:
      return "[DEBUG] ";
    case LogSeverity::kVERBOSE:
      return "[VERBOSE] ";
    default:
      return "[UNKNOWN] ";
    }
  };

  std::string prefix = severity_to_string(severity);

  std::cerr << prefix;
  if (color_) {
    std::cerr << LLMX_COLOR_NORMAL;
  }
  std::cerr << msg << '\n';

  if (!path.empty()) {
    try {
      std::ofstream ofs(path, std::ios::app);
      if (ofs.is_open()) {
        ofs << prefix << msg << '\n';
      }
    } catch (...) {
    }
  }
}

Logger::LogSeverity Logger::get_reportable_severity() const noexcept {
  return severity_;
}

void Logger::set_reportable_log_severity(LogSeverity severity) noexcept {
  severity_ = severity;
}

void Logger::set_log_color(bool color) noexcept { color_ = color; }

Logger &get_logger() noexcept {
  static Logger logger{Logger::LogSeverity::kDEBUG, true};
  return logger;
}

} // namespace llmx