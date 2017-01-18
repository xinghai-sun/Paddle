/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <glog/logging.h>
#include <stdio.h>
#include <memory>
#include <string>
#include "Compiler.h"

namespace paddle {

/**
 * Error is Paddle error code. It only contain a std::string as error message.
 *
 *
 * There are two styles to return error in Paddle.
 *
 * 1. Return Error
 *    When method return a status, the return must use `__must_check` attribute.
 *    Example as below.
 * @code{cpp}
 * Error __must_check foo();
 *
 * Error __must_check bar() {
 *   // do something.
 *   Status s = foo();  // invoke other method return status.
 *   if (!s) return s;
 *   // do something else.
 *   return Status();
 * }
 * @endcode{cpp}
 *
 * 2. Return by parameter.
 *    It is another way to return an error, by using a pointer parameter.
 *    Example as below.
 *
 * @code{cpp}
 * Error bar();
 *
 * int foo(Error* error) {
 *   // Do something.
 *   Error s = bar();
 *   if (!s) {
 *     *error = s;
 *     return 0;
 *   }
 *   // Do something else.
 *   if (someInternalErrorHappend) {
 *     *error = Error("Some dimension is too large, %d", dimension);
 *     return 0;
 *   }
 *   // End of method.
 *   return someValue;
 * }
 *
 * Error foobar() {
 *   Error s;
 *   // do something.
 *   foo(&s);
 *   if (!s) return s;
 * }
 * @endcode{cpp}
 *
 *
 * Currently there is a helper method 'check' in status, because Paddle always
 * use log(FATAL) or CHECK to make program exit before. When we clean all
 * log(FATAL) and CHECK in Paddle, 'check' method will be removed.
 */
class Error final {
public:
  /**
   * Default Status. OK
   */
  inline Error() {}

  /**
   * @brief Create an Error use printf syntax.
   */
  inline explicit Error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    constexpr size_t kBufferSize = 1024;
    this->errMsg_.reset(new std::string(kBufferSize, 0));
    auto sz = vsnprintf(&(*errMsg_)[0], kBufferSize, fmt, ap);
    this->errMsg_->resize(sz);
    this->errMsg_->shrink_to_fit();
    va_end(ap);
  }

  /**
   * @brief what will return the error message. If no error, return nullptr.
   */
  inline const char* msg() const {
    if (errMsg_) {
      return errMsg_->data();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief operator bool, return True if there is no error.
   */
  inline operator bool() const { return !errMsg_; }
  /**
   * @brief check this status by glog.
   * @note It is a temp method used during cleaning Paddle code. It will be
   *       removed later.
   */
  inline void check() const { CHECK(*this) << msg(); }

private:
  std::shared_ptr<std::string> errMsg_;
};

}  // namespace paddle
