// Copyright 2013 Tetsuo Kiso. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef KERNEL_PA_LEARNER_H_
#define KERNEL_PA_LEARNER_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include "common.h"

namespace kernel {

const double kC = 1.0;
const int kKernelDegree = 2;

// Learner implements kernelized passive aggressive (PA-I) algorithm
// (Crammer et al., JMLR 2006).
//
// We only support polynomial kernel function, which
// is often used for training natural language applications.
//
// Because kernel computation is slow, we use a polynomial kernel
// inverted method (PKI) (Kudo and Matsumoto, ACL 2003) for computing
// a polynomial kernel function efficiently.
//
// The time complexity of PKI is O(B|x| + |S|) where B is the average
// of |sv_index_[i]| over all feature vectors, |x| is the number of
// active features in a feature vector x, and |S| is the size of
// support set S.
class Learner {
 public:
  Learner();
  virtual ~Learner();

  bool Train(const char* filename, int iter);

  // Read a training data.
  bool Read(const char* train_file,
            std::vector<example>& examples,
            std::size_t* maxid);

  // Load a trained model.
  bool Load(const char* filename);

  // Save a trained model to a disk.
  bool Save(const char* filename) const;

  // Set the degree of a polynomial kernel you want to use.
  void SetKernelDegree(int d) {
    d_ = d;
  }

  // Set the hyperparameter of PA-I.
  void SetC(float C) {
    C_ = C;
  }

  short Predict(const fv& x) {
    const float m = Margin(x);
    return (m >= 0.0) ? 1 : -1;
  }

  // Compute margin with the PKI.
  float Margin(const fv& x) {
    ClearMargin();

    for (const auto& p : x) {
      const unsigned int id = p.first;
      const float v = p.second;
      if (id >= sv_index_.size()) {
        continue;
      }
      const fv& s = sv_index_[id];
      for (const auto& p2 : s) {
        margin_[p2.first] += p2.second * v;
      }
    }

    return Polynomial();
  }

  float HingeLoss(float m, short y) const {
    return std::max(0.0f, 1.0f - y * m);
  }

  float LearningRate(float loss, const fv& x) const {
    const float norm = L2Norm(x);
    return std::min(C_, loss / norm);
  }

  void UpdateAlpha(const fv& x, float rate, short y) {
    alpha_.push_back(y * rate);
    margin_.push_back(0.0f);
  }

  void UpdateIndex(const fv& x, unsigned int sv_id) {
    for (const auto& p : x) {
      const unsigned int id = p.first;
      if (sv_index_.size() <= id) {
        sv_index_.resize(id + 1);
      }
      sv_index_[id].push_back(std::make_pair(sv_id, p.second));
    }
  }

 private:
  // Compute polynomial kernel function.
  float Polynomial() const;

  void ClearMargin() {
    for (std::size_t i = 0; i < margin_.size(); ++i) {
      margin_[i] = 0.0f;
    }
  }

  // hyperparameter of passive-aggressive.
  float C_;

  // the degree of polynomial kernel
  int d_;

  std::vector<float> alpha_;

  // The inverted index for computing the polynomial kernel
  // efficiently.
  std::vector<fv> sv_index_;

  // Used for computing margin.
  //
  // The size of this vector is equal to the number of support
  // vectors (size of the support set).
  std::vector<float> margin_;

  // We disable the default copy constructor and assignment operator.
  Learner(const Learner&);
  void operator=(const Learner&);
};
} // namespace kernel
#endif // KERNEL_PA_LEARNER_H_
