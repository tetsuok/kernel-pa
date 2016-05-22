// Copyright 2013 Tetsuo Kiso. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <string>
#include <unistd.h>

#include "learner.h"
#include "logging.h"

using namespace std;

namespace kernel {
namespace {

// the default number of iterations.
const int kNumIter = 3;

// the default model filename.
const char kModel[] = "model";

void Usage(const char* name) {
  cerr << "usage: " << name << " [-C FLOAT] [-d INT] [-t INT] [-o FILE] FILE\n"
      "-C hyperparameter passive-aggressive. C should be positive. (default: " << kC << ")\n"
      "-d the degree of polynomial kernel (supported: d = 1, 2, 3, and 4) (deafult: " << kKernelDegree << ")\n"
      "-t the number of iterations (default: " << kNumIter << ")\n"
      "-o model filename (default " << kModel << ")\n";
  exit(1);
}

} // namespace
} // namespace kernel

int main(int argc, char** argv) {
  float C = kernel::kC;
  int kernel_degree = kernel::kKernelDegree;
  int iter = kernel::kNumIter;
  string model = kernel::kModel;

  int opt;
  while ((opt = getopt(argc, argv, ":C:d:t:o:")) != -1) {
    switch (opt) {
      case 'C':
        C = atof(optarg);
        break;
      case 'd':
        kernel_degree = atoi(optarg);
        break;
      case 't':
        iter = atoi(optarg);
        break;
      case 'o':
        model = string(optarg);
        break;
      default:
        kernel::Usage(argv[0]);
    }
  }

  if (optind + 1 != argc) {
    kernel::Usage(argv[0]);
  }

  kernel::Learner learner;
  learner.SetC(C);
  learner.SetKernelDegree(kernel_degree);

  if (!learner.Train(argv[optind], iter)) {
    LOG(ERROR) << "cannot train " << argv[optind];
    return -1;
  }
  if (!learner.Save(model.c_str())) {
    LOG(ERROR) << "cannot save trained model " << model;
  }
  return 0;
}
