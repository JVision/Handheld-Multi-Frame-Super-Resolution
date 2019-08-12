#pragma once

#include <opencv2/core.hpp>
#include <numeric>

template<typename  T>
cv::Mat getKernelWeights(int rows, int cols, T a, T b, T c, T d)
{
    const auto x_mid = (cols-1 ) / 2.0;
    const auto y_mid = (rows-1 ) / 2.0;

    auto kernel = cv::Mat_<T>(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            auto y = i - y_mid;
            auto x = j - x_mid;
            kernel.ptr<T>(i)[j] = std::exp(-0.5* (a*x*x + (b + c)*x*y + d * y*y));
        }
    }
    //Scalar t = sum(sum(kernel));

    const auto denominator = 1.0f;
    //std::accumulate(kernel.begin(), kernel.end(), 0.0);
    return kernel / denominator;
}
template<typename  T>
cv::Mat getKernelWeights(int rows, int cols, cv::Vec<T, 4> & omega)
{
    return getKernelWeights(rows, cols, omega[0], omega[2], omega[1], omega[3]);
}