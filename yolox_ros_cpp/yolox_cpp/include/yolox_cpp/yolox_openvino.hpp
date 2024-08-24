#ifndef _YOLOX_CPP_YOLOX_OPENVINO_HPP
#define _YOLOX_CPP_YOLOX_OPENVINO_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_cpp{
    class YoloXOpenVINO: public AbcYoloX{
        public:
            YoloXOpenVINO(const file_name_t &path_to_model, std::string device_name,
                          float nms_th=0.45, float conf_th=0.3, const std::string &model_version="0.1.1rc0",
                          int num_classes=80, bool p6=false);
            std::vector<Object> inference(const cv::Mat& frame) override;

        private:
            std::string device_name_;
            std::vector<float> blob_;
            ov::Shape input_shape_;
            ov::InferRequest infer_request_;
    };
}

#endif
