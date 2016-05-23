// Copyright 2013 Tetsuo Kiso. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include "logging.h"
#include "learner.h"

namespace kernel {
namespace {

class Result {
 public:
  bool Add(short y, short predict) {
    if      (y ==  1 && predict ==  1) { results_[0]++; }
    else if (y == -1 && predict == -1) { results_[1]++; }
    else if (y == -1 && predict ==  1) { ++mistake_; results_[2]++; }
    else if (y ==  1 && predict == -1) { ++mistake_; results_[3]++; }
    else                               { return false;  }

    ++num_instance_;

    return true;
  }

  void Show() const {
    std::printf("Accuracy %.3f%% (%d/%d)\n(Answer, Predict): "
                "(t,p):%d (t,n):%d (f,p):%d (f,n):%d\n",
                CalcAccuracy(), num_instance_ - mistake_, num_instance_,
                results_[0], results_[1], results_[2], results_[3]);
  }

  double CalcAccuracy() const {
    return static_cast<double>((num_instance_ - mistake_) * 100.0 / num_instance_);
  }

  unsigned int get_true_positive() const { return results_[0]; }
  unsigned int get_true_negative() const { return results_[1]; }
  unsigned int get_false_positive() const { return results_[2]; }
  unsigned int get_false_negative() const { return results_[3]; }

  unsigned int get_num_instance() const { return num_instance_; }

  unsigned int get_mistake() const { return mistake_; }

 private:
  unsigned int num_instance_ = 0;         // Number of classified instance
  unsigned int mistake_ = 0;              // Number of mistakes
  unsigned int results_[4] = {0};         // results of classification
};

int KernelPAClassify(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s test_file model\n", argv[0]);
    return -1;
  }

  const char* test_file = argv[1];
  const char* model = argv[2];
  kernel::Learner learner;
  if (!learner.Load(model)) {
    LOG(ERROR) << "cannot load " << model;
    return -1;
  }

  std::ifstream ifs(test_file);
  if (!ifs) {
    LOG(ERROR) << "no such file or directory " << test_file;
    return -1;
  }

  Result result;
  unsigned int line_num = 0;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line[0] == '#' || line.empty()) continue;       // ignore comments
    ++line_num;

    short label;
    fv x;
    size_t dummy = 0;
    if (!Tokenize(line.c_str(), &x, &label, &dummy)) {
      LOG(ERROR) << "cannot tokenize: " << line;
      return -1;
    }

    const short predict = learner.Predict(x);
    result.Add(label, predict);
  }

  result.Show();
  return 0;
}
} // namespace
} // namespace kernel

int main(int argc, char** argv) {
  return kernel::KernelPAClassify(argc, argv);
}
