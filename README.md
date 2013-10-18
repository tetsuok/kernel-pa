kernel-pa
=========

An implementation of kernelized passive-aggressive algorithm (PA-I) [1].
This tool is created particularly for natural language processing applications.

## Features

- Fast kernel-based training (using the PKI method [2])
- Practical accuracy with online kernel passive-aggressive algorithm
- Compact design, written in C++ with STL
- Support learning with polynomial kernel function.

Note that we do not support shuffling the training data. Users may
want to shuffle the data before using this tool.

## Software Requirements

- GNU C++ compiler or Apple's Clang

We have tested our code on Ubuntu Linux 12.04 (x86_64) and OS X 10.8.5.

## Compilation

    $ make

## Training

    $ ./kernel_pa [-C FLOAT] [-d INT] [-t INT] [-o FILE] train_file

The data format of the training and test data is the same as [SVMLight](http://svmlight.joachims.org/).

### Options

- -C hyperparameter of passive-aggressive algorithm
- -d the degree of polynomial kernel function. We support: d = 1, 2, 3, and 4.
- -t the number of iterations
- -o model file

## Testing

    $ ./kernel_pa_classify test_file model_file

## References

[1] Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, and Yoram Singer. Online passive-aggressive algorithms. Journal of Machine Learning Research, 7:551–585.

[2] Taku Kudo and Yuji Matsumoto. Fast methods for kernel-based text analysis. In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics (ACL). 2003. pages 24-31.

## Acknowledgements

Thanks for Daisuke Okanohara. His online learning toolkit, [oll](https://code.google.com/p/oll/) is a good resource to make sure implemented code works fine.

## License

This code is distributed under the New BSD License. See the file LICENSE.
