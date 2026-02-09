#ifndef LINESEARCH_HPP
#define LINESEARCH_HPP

#include <Eigen>

template<int N>
class LineSearch {
private:
    float f;
    Eigen::Matrix<N, 1, 1> gradient;
    Eigen::Matrix<N, N, 1> hessian;
};

#endif