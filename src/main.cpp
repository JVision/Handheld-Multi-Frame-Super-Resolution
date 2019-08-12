#include "kernels.h"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

/* this function implement the heuristic part on the paper:                                            */
/* that small dominant eigenvalues (comparable to the amount of noise expected in the given raw image) */
/* signify relatively flat, noisy regions while large eigenvalues appear around features whose         */
/* resolution we want to enhance. */
/*
Parameters:
    D_th and D_tr are in units of gradient magnitude of the image normalized to the range.
    usual range,  D_th = [0.001, ..., 0.010],
    usual range,, D_tr = [0.006, ..., 0.020],
    k_stretch as the amount of kernel stretching along the edges
    k_shrink as the amount of kernel shrinking perpendicular to the edges
    k_detail as the base kernel standard deviation
    k_denoise as the kernel standard deviation suitable for denoising
*/

void getKernelPara(float lambda1, float lambda2, float D_tr, float D_th, float &k1, float &k2, float k_detail, float k_denoise, float K_shrink, float k_streach)
{
    //auto A = 1 + sqrt((lambda1 - lambda2) / (lambda1 + lambda2));
    auto A = 1 + sqrt((lambda1 ) / (lambda2));
    auto D = std::min(std::max(1.0 - sqrt(lambda1) / D_tr + D_th, 0.0), 1.0);
    auto k_1_hat = k_detail * k_streach * A;
    auto k_2_hat = k_detail / K_shrink / A;
    k1 = ((1 - D)*k_1_hat + D * k_detail*k_denoise);
    k2 = ((1 - D)*k_2_hat + D * k_detail*k_denoise);
    k1 *= k1;
    k2 *= k2;
}


Mat testKernelGeneration()
{
    cv::Mat e = cv::Mat::eye(2, 2, CV_32F);

    float lambda1 = 0.1, lambda2 = lambda1/6.5*4.5;
    float k1, k2;
    float D_tr = 0.01, D_th = 0.005;
    float k_detail = 0.29f, k_denoise = 4.0f, K_shrink = 2.0f, k_streach = 4.0f;
    getKernelPara(lambda1, lambda2, D_tr, D_th, k1, k2, k_detail, k_denoise, K_shrink, k_streach);
    Mat k(2, 2, CV_32F, cv::Scalar::all(0));
    Mat omega(2, 2, CV_32F);

    typedef float T;

    bool inverse = true;
    {
        k.at<T>(0) = k1;
        k.at<T>(3) = k2;
        omega = e * k*e.t();
    }
    if (inverse)
    {
        omega = omega.inv();
    }

    Vec<T, 4> o;
    for (int i = 0; i < 4; i++) {
        o[i] = omega.at<float>(i);
    }

    Mat weights = getKernelWeights(5, 5, o);
    return weights;
}



int main(void)
{
	Mat weights = testKernelGeneration();
	cout << weights << std::endl;
}