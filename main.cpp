#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "model.h"



int main(){
    DSK::TEST FR;

    // to test a function
    FR.check();

    cv::Mat Image = cv::imread("../2.jpg");

    std::vector<float> results = FR.tmain_old(Image);

    for (auto x : results){
        std:: cout << x << ",";
    }
}