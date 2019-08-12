#include "kernels.h"
using namespace cv;
using namespace std;

void testKernelGeneration()
{
    cv::Mat e = cv::Mat::eye(2, 2, CV_32F);

    float lambda1 = 0.01, lambda2 = lambda1/19.0;
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
}



int main(void)
{
	testKernelGeneration();
	cout << weights << std::endl;
	
}