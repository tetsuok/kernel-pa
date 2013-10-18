// Copyright 2013 Tetsuo Kiso. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef KERNEL_PA_TOKENIZER_H_
#define KERNEL_PA_TOKENIZER_H_

#include <cstddef>
#include "common.h"

namespace kernel {

class Tokenizer {
 public:
  static bool Tokenize(const char* line, fv* fv, short* y,
                       std::size_t* maxid);

 private:
  Tokenizer();
  virtual ~Tokenizer();
};
} // namespace kernel
#endif  // KERNEL_PA_TOKENIZER_H_
