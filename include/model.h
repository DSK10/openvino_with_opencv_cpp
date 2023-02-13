#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <dirent.h>



// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
// #include "samples/slog.hpp"
#include "opencv2/opencv.hpp"

namespace DSK {
    class TEST {
        public:
            void check();
            std::vector<float> tmain_old(cv::Mat MatImage);
    };
}