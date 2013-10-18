// Copyright 2013 Tetsuo Kiso. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef KERNEL_PA_LOGGING_H_
#define KERNEL_PA_LOGGING_H_

#include <cstdlib>
#include <fstream>
#include <iostream>

namespace logging {

enum LogLevel {
  LOG_INFO = 0,
  LOG_WARNING = 1,
  LOG_ERROR = 2,
  LOG_FATAL = 3
};

class Logger {
 public:
  static std::ostream &GetLogStream() {
    return std::cerr;
  }

 private:
  Logger() {}
  virtual ~Logger() {}
};

class LogFinalizer {
 public:
  explicit LogFinalizer(LogLevel level) : level_(level) {}
  ~LogFinalizer() {
    Logger::GetLogStream() << std::endl;
    if (level_ == LOG_FATAL) {
      std::exit(-1);
    }
  }

  // To ignore values used in when defining logging macros.
  void operator&(std::ostream&) {}

 private:
  LogLevel level_;
};
} // namespace logging

#define CHECK_AND_DIE(condition) \
  (condition) ? (void) 0 : logging::LogFinalizer(logging::LOG_FATAL) & \
  logging::Logger::GetLogStream() << "Die: " << \
  __FILE__ << "(" << __LINE__ << ") [" << #condition << "] "

#define LOG(loglevel) \
  logging::LogFinalizer(logging::LOG_##loglevel) & \
  logging::Logger::GetLogStream() \
  << "LOG(" <<  #loglevel << "): " << __FILE__ << "(" << __LINE__ << ") "

#endif  // KERNEL_PA_LOGGING_H_
