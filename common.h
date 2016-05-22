// Copyright 2013 Tetsuo Kiso. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef KERNEL_PA_COMMON_H_
#define KERNEL_PA_COMMON_H_

#include <vector>
#include <utility>

namespace kernel {

// feature vector
typedef std::vector<std::pair<unsigned int, float> > fv;

// training example (x, y)
typedef std::pair<fv, short> example;

// Compute |x|^2
inline float L2Norm(const fv& x) {
  float r = 1.0f; // bias
  for (size_t i = 0; i < x.size(); ++i) {
    r += x[i].second * x[i].second;
  }
  return r;
}

bool Tokenize(const char *line, fv *fv, short *y, std::size_t *maxid);

} // namespace kernel
#endif // KERNEL_PA_COMMON_H_
