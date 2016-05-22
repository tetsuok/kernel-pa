// Copyright 2013 Tetsuo Kiso. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "learner.h"

#include <fstream>
#include "logging.h"

using namespace std;

namespace kernel {

Learner::Learner() : C_(kC), d_(kKernelDegree) {}
Learner::~Learner() {}

bool Learner::Read(const char* train_file,
                   vector<example>& examples,
                   size_t* maxid) {
  ifstream ifs(train_file);
  if (!ifs) {
    LOG(ERROR) << "no such file or directory " << train_file;
    return false;
  }

  string line;
  size_t line_num = 0;
  while (std::getline(ifs, line)) {
    ++line_num;
    if (line[0] == '#' || line.empty()) continue;

    short label = 0;                      // true label
    fv vec;
    if (!Tokenize(line.c_str(), &vec, &label, maxid)) {
      LOG(ERROR) << "Invalid line: " << line_num;
      return false;
    }
    examples.push_back(make_pair(vec, label));
  }
  return true;
}

bool Learner::Train(const char* filename, int iter) {
  vector<example> examples;
  size_t maxid = 0;
  if (!Read(filename, examples, &maxid)) {
    LOG(ERROR) << "cannot read training data " << filename;
    return false;
  }

  const std::size_t num_examples = examples.size();

  for (int t = 0; t < iter; ++t) {
    for (size_t i = 0; i < num_examples; ++i) {
      fprintf(stderr, "iter = %d: (%lu/%lu)\r", t+1, i, num_examples);

      const float m = Margin(examples[i].first);
      const float loss = HingeLoss(m, examples[i].second);
      if (loss >= 0.0) {
        const float rate = LearningRate(loss, examples[i].first);
        UpdateIndex(examples[i].first,
                    static_cast<unsigned int>(alpha_.size()));
        UpdateAlpha(examples[i].first, rate, examples[i].second);
      }
    }
  }

  fprintf(stderr, "INFO: the number of support vectors = %lu\n",
          margin_.size());
  return true;
}

bool Learner::Load(const char* filename) {
  std::ifstream bifs(filename, std::ios::in | std::ios::binary);
  if (!bifs) {
    LOG(ERROR) << "Failed to load " << filename;
    return false;
  }

  bifs.read(reinterpret_cast<char *>(&C_), sizeof(C_));
  bifs.read(reinterpret_cast<char *>(&d_), sizeof(d_));

  std::size_t alpha_size;
  bifs.read(reinterpret_cast<char *>(&alpha_size), sizeof(alpha_size));
  alpha_.resize(alpha_size);
  bifs.read(reinterpret_cast<char *>(&alpha_[0]), alpha_.size() * sizeof(float));

  std::size_t sv_size;
  bifs.read(reinterpret_cast<char *>(&sv_size), sizeof(sv_size));
  sv_index_.resize(sv_size);

  for (size_t i = 0; i < sv_index_.size(); ++i) {
    fv& x = sv_index_[i];
    std::size_t x_size;
    bifs.read(reinterpret_cast<char *>(&x_size), sizeof(x_size));
    x.resize(x_size);
    for (size_t j = 0; j < x.size(); ++j) {
      bifs.read(reinterpret_cast<char *>(&x[j].first), sizeof(x[j].first));
      bifs.read(reinterpret_cast<char *>(&x[j].second), sizeof(x[j].second));
    }
  }

  margin_.resize(alpha_.size(), 0.0f);

  bifs.close();
  return true;
}

bool Learner::Save(const char* filename) const {
  std::ofstream bofs;
  bofs.open(filename, ios::out | ios::binary);
  if (!bofs) {
    LOG(ERROR) << "no such file or directory: " << filename;
    return false;
  }

  bofs.write(reinterpret_cast<const char *>(&C_), sizeof(C_));
  bofs.write(reinterpret_cast<const char *>(&d_), sizeof(d_));

  const std::size_t alpha_size = alpha_.size();
  bofs.write(reinterpret_cast<const char *>(&alpha_size), sizeof(alpha_size));
  bofs.write(reinterpret_cast<const char *>(&alpha_[0]), alpha_.size() * sizeof(float));

  const std::size_t sv_size = sv_index_.size();
  bofs.write(reinterpret_cast<const char *>(&sv_size), sizeof(sv_size));

  for (size_t i = 0; i < sv_index_.size(); ++i) {
    const fv& x = sv_index_[i];
    const std::size_t x_size = x.size();
    bofs.write(reinterpret_cast<const char *>(&x_size), sizeof(x_size));

    for (size_t j = 0; j < x.size(); ++j) {
      bofs.write(reinterpret_cast<const char *>(&x[j].first), sizeof(x[j].first));
      bofs.write(reinterpret_cast<const char *>(&x[j].second), sizeof(x[j].second));
    }
  }

  bofs.close();
  return true;
}

// This is a bit ugly.
// We support only d = 1, 2, 3, and 4.
float Learner::Polynomial() const {
  float r = 0.0f;

  if (d_ == 1) {
    for (size_t i = 0; i < margin_.size(); ++i) {
      r += margin_[i] * alpha_[i];
    }
  } else if (d_ == 2) {
    for (size_t i = 0; i < margin_.size(); ++i) {
      r += (margin_[i] * margin_[i]) * alpha_[i];
    }
  } else if (d_ == 3) {
    for (size_t i = 0; i < margin_.size(); ++i) {
      r += (margin_[i] * margin_[i] * margin_[i]) * alpha_[i];
    }
  } else if (d_ == 4) {
    for (size_t i = 0; i < margin_.size(); ++i) {
      r += (margin_[i] * margin_[i] * margin_[i] * margin_[i]) * alpha_[i];
    }
  }
  return r;
}

} // namespace kernel
