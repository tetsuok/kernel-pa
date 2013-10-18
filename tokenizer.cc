// Copyright 2013 Tetsuo Kiso. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tokenizer.h"
#include "logging.h"

namespace kernel {

bool Tokenizer::Tokenize(const char* line, fv* fv, short* y,
                         std::size_t* maxid) {
  if (line[0] == '#') return false;

  *y = static_cast<short>(atoi(line));
  if (*y != 1 && *y != -1) {
    LOG(ERROR) << "Invalid label. A label must be +1 or -1.";
    return false;
  }

  while (!isspace(*line) && *line) line++;
  while (isspace(*line) && *line) line++;
  while (1) {
    const char *begin = line;
    unsigned int index = std::atol(begin);
    while (*line != ':' && *line) line++;
    float val = std::atof(++line);

    while (*line != ' ' && *line) line++;
    fv->push_back(std::make_pair(index, val));
    if (index > *maxid) *maxid = index;
    if (*line++ == 0) break;
  }
  return true;
}
} // namespace kernel
