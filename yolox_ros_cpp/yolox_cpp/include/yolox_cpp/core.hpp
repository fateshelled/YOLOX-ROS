#ifndef _YOLOX_CPP_CORE_HPP
#define _YOLOX_CPP_CORE_HPP

#include <opencv2/core/types.hpp>
#include <opencv2/core/simd_intrinsics.hpp>
#include <algorithm>

namespace yolox_cpp
{
/**
 * @brief Define names based depends on Unicode path support
 */
#define file_name_t std::string

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };

    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
        GridAndStride(const int grid0_, const int grid1_, const int stride_)
            : grid0(grid0_), grid1(grid1_), stride(stride_)
        {
        }
    };

    class AbcYoloX
    {
    public:
        AbcYoloX() {}
        AbcYoloX(const float nms_th = 0.45, const float conf_th = 0.3,
                 const std::string &model_version = "0.1.1rc0",
                 const int num_classes = 80, const bool p6 = false)
            : nms_thresh_(nms_th), bbox_conf_thresh_(conf_th),
              num_classes_(num_classes), p6_(p6), model_version_(model_version)
        {
        }
        virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

    protected:
        int input_w_;
        int input_h_;
        float nms_thresh_;
        float bbox_conf_thresh_;
        int num_classes_;
        bool p6_;
        std::string model_version_;
        // const std::vector<float> mean_ = {0.485, 0.456, 0.406};
        // const std::vector<float> std_ = {0.229, 0.224, 0.225};
        const std::vector<float> std255_inv_ = {
            1.0 / (255.0 * 0.229), 1.0 / (255.0 * 0.224), 1.0 / (255.0 * 0.225)};
        const std::vector<float> mean_std_ = {
            -0.485 / 0.229 , -0.456 / 0.224, -0.406 / 0.225};
        const std::vector<int> strides_ = {8, 16, 32};
        const std::vector<int> strides_p6_ = {8, 16, 32, 64};
        std::vector<GridAndStride> grid_strides_;

        cv::Mat static_resize(const cv::Mat &img)
        {
            const float r = std::min(
                static_cast<float>(input_w_) / (static_cast<float>(img.cols) * 1.0f),
                static_cast<float>(input_h_) / (static_cast<float>(img.rows) * 1.0f));
            // r = std::min(r, 1.0f);
            const int unpad_w = r * img.cols;
            const int unpad_h = r * img.rows;
            cv::Mat re(unpad_h, unpad_w, CV_8UC3);
            cv::resize(img, re, re.size());
            cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
            re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
            return out;
        }

        // for NCHW
        void blobFromImage(const cv::Mat &img, float *blob_data)
        {
            blobFromImage_cpu(img, blob_data);
            // #if defined(CV_SIMD128) && CV_SIMD128 == 1
            //     blobFromImage_simd(img, blob_data);
            // #else
            //     blobFromImage_cpu(img, blob_data);
            // #endif
        }

        // for NHWC
        void blobFromImage_nhwc(const cv::Mat &img, float *blob_data)
        {
            blobFromImage_nhwc_cpu(img, blob_data);
            // #if defined(CV_SIMD128) && CV_SIMD128 == 1
            //     blobFromImage_nhwc_simd(img, blob_data);
            // #else
            //     blobFromImage_nhwc_cpu(img, blob_data);
            // #endif
        }

        void generate_grids_and_stride(const int target_w, const int target_h, const std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
        {
            grid_strides.clear();
            for (auto stride : strides)
            {
                const int num_grid_w = target_w / stride;
                const int num_grid_h = target_h / stride;
                for (int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        grid_strides.emplace_back(g0, g1, stride);
                    }
                }
            }
        }

        void generate_yolox_proposals(const std::vector<GridAndStride> &grid_strides, const float *feat_ptr, const float prob_threshold, std::vector<Object> &objects)
        {
            const int num_anchors = grid_strides.size();
            objects.clear();

            for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx)
            {
                const int grid0 = grid_strides[anchor_idx].grid0;
                const int grid1 = grid_strides[anchor_idx].grid1;
                const int stride = grid_strides[anchor_idx].stride;

                const int basic_pos = anchor_idx * (num_classes_ + 5);

                const float box_objectness = feat_ptr[basic_pos + 4];
                int class_id = 0;
                float max_class_score = 0.0;
                for (int class_idx = 0; class_idx < num_classes_; ++class_idx)
                {
                    const float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
                    const float box_prob = box_objectness * box_cls_score;
                    if (box_prob > max_class_score)
                    {
                        class_id = class_idx;
                        max_class_score = box_prob;
                    }
                }
                if (max_class_score > prob_threshold)
                {
                    // yolox/models/yolo_head.py decode logic
                    //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
                    //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
                    const float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
                    const float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
                    const float w = exp(feat_ptr[basic_pos + 2]) * stride;
                    const float h = exp(feat_ptr[basic_pos + 3]) * stride;
                    const float x0 = x_center - w * 0.5f;
                    const float y0 = y_center - h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = w;
                    obj.rect.height = h;
                    obj.label = class_id;
                    obj.prob = max_class_score;
                    objects.push_back(obj);
                }
            } // point anchor loop
        }

        float intersection_area(const Object &a, const Object &b)
        {
            const cv::Rect_<float> inter = a.rect & b.rect;
            return inter.area();
        }

        void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, const float nms_threshold)
        {
            picked.clear();

            const int n = faceobjects.size();

            std::vector<float> areas(n);
            for (int i = 0; i < n; ++i)
            {
                areas[i] = faceobjects[i].rect.area();
            }

            for (int i = 0; i < n; ++i)
            {
                const Object &a = faceobjects[i];
                const int picked_size = picked.size();

                int keep = 1;
                for (int j = 0; j < picked_size; ++j)
                {
                    const Object &b = faceobjects[picked[j]];

                    // intersection over union
                    const float inter_area = intersection_area(a, b);
                    const float union_area = areas[i] + areas[picked[j]] - inter_area;
                    // float IoU = inter_area / union_area
                    if (inter_area / union_area > nms_threshold)
                        keep = 0;
                }

                if (keep)
                    picked.push_back(i);
            }
        }

        void decode_outputs(const float *prob, const std::vector<GridAndStride> &grid_strides,
                            std::vector<Object> &objects, const float bbox_conf_thresh,
                            const float scale, const int img_w, const int img_h)
        {

            std::vector<Object> proposals;
            generate_yolox_proposals(grid_strides, prob, bbox_conf_thresh, proposals);

            std::sort(
                proposals.begin(), proposals.end(),
                [](const Object& a, const Object& b) {
                    return a.prob > b.prob; // descent
                }
            );

            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_thresh_);

            int count = picked.size();
            objects.resize(count);
            const float max_x = static_cast<float>(img_w - 1);
            const float max_y = static_cast<float>(img_h - 1);

            for (int i = 0; i < count; ++i)
            {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = static_cast<float>(objects[i].rect.x) / scale;
                float y0 = static_cast<float>(objects[i].rect.y) / scale;
                float x1 = static_cast<float>(objects[i].rect.x + objects[i].rect.width) / scale;
                float y1 = static_cast<float>(objects[i].rect.y + objects[i].rect.height) / scale;

                // clip
                x0 = std::max(std::min(x0, max_x), 0.f);
                y0 = std::max(std::min(y0, max_y), 0.f);
                x1 = std::max(std::min(x1, max_x), 0.f);
                y1 = std::max(std::min(y1, max_y), 0.f);

                objects[i].rect.x = x0;
                objects[i].rect.y = y0;
                objects[i].rect.width = x1 - x0;
                objects[i].rect.height = y1 - y0;
            }
        }

    private:
    #if defined(CV_SIMD128) && CV_SIMD128 == 1
        void blobFromImage_simd(const cv::Mat &img, float *blob_data)
        {
            const size_t channels = 3;
            const size_t img_h = img.rows;
            const size_t img_w = img.cols;
            const size_t img_hw = img_h * img_w;

            const size_t step = 4; // load 4 pixel
            const size_t N = img_hw / step;
            const size_t remain = img_hw % step;

            float *blob_data_ch0 = blob_data;
            float *blob_data_ch1 = blob_data + img_hw;
            float *blob_data_ch2 = blob_data + img_hw * 2;

            if (this->model_version_ == "0.1.0")
            {
                cv::Mat img_f32;
                img.convertTo(img_f32, CV_32FC3);
                const cv::v_float32x4 mean_std0 = cv::v_setall_f32(-this->mean_std_[0]);
                const cv::v_float32x4 mean_std1 = cv::v_setall_f32(-this->mean_std_[1]);
                const cv::v_float32x4 mean_std2 = cv::v_setall_f32(-this->mean_std_[2]);
                const cv::v_float32x4 std255_inv_0 = cv::v_setall_f32(this->std255_inv_[0]);
                const cv::v_float32x4 std255_inv_1 = cv::v_setall_f32(this->std255_inv_[1]);
                const cv::v_float32x4 std255_inv_2 = cv::v_setall_f32(this->std255_inv_[2]);

                for (size_t i = 0; i < N; ++i)
                {
                    cv::v_float32x4 ch0_f;
                    cv::v_float32x4 ch1_f;
                    cv::v_float32x4 ch2_f;
                    // load 4 pixel x 3ch
                    cv::v_load_deinterleave(
                        reinterpret_cast<const float*>(img_f32.data) + i * (step * channels),
                        ch0_f, ch1_f, ch2_f);

                    {
                        ch0_f = ch0_f * std255_inv_0 + mean_std0;
                        ch1_f = ch1_f * std255_inv_1 + mean_std1;
                        ch2_f = ch2_f * std255_inv_2 + mean_std2;
                    }

                    cv::v_store(blob_data_ch0 + i * step, ch0_f);
                    cv::v_store(blob_data_ch1 + i * step, ch1_f);
                    cv::v_store(blob_data_ch2 + i * step, ch2_f);
                }
            }
            else
            {
                cv::Mat img_f32;
                img.convertTo(img_f32, CV_32FC3);
                for (size_t i = 0; i < N; ++i)
                {
                    cv::v_float32x4 ch0_f;
                    cv::v_float32x4 ch1_f;
                    cv::v_float32x4 ch2_f;
                    // load 4 pixel x 3ch
                    cv::v_load_deinterleave(
                        reinterpret_cast<const float*>(img_f32.data) + i * (step * channels),
                        ch0_f, ch1_f, ch2_f);

                    cv::v_store(blob_data_ch0 + i * step, ch0_f);
                    cv::v_store(blob_data_ch1 + i * step, ch1_f);
                    cv::v_store(blob_data_ch2 + i * step, ch2_f);
                }
            }

            if (remain > 0)
            {
                const size_t simd_done_num = N * step;
                if (this->model_version_ == "0.1.0")
                {
                    for (size_t i = 0; i < remain; ++i)
                    {
                        // HWC -> CHW
                        const size_t out_idx = simd_done_num + i;
                        const size_t src_idx = out_idx * channels;
                        blob_data_ch0[out_idx] = static_cast<float>(img.data[src_idx + 0]) * this->std255_inv_[0] + this->mean_std_[0];
                        blob_data_ch1[out_idx] = static_cast<float>(img.data[src_idx + 1]) * this->std255_inv_[1] + this->mean_std_[1];
                        blob_data_ch2[out_idx] = static_cast<float>(img.data[src_idx + 2]) * this->std255_inv_[2] + this->mean_std_[2];
                    }
                }
                else
                {
                    for (size_t i = 0; i < remain; ++i)
                    {
                        // HWC -> CHW
                        const size_t out_idx = simd_done_num + i;
                        const size_t src_idx = out_idx * channels;
                        blob_data_ch0[out_idx] = static_cast<float>(img.data[src_idx + 0]);
                        blob_data_ch1[out_idx] = static_cast<float>(img.data[src_idx + 1]);
                        blob_data_ch2[out_idx] = static_cast<float>(img.data[src_idx + 2]);
                    }

                }
            }

        }
    #endif
        void blobFromImage_cpu(const cv::Mat &img, float *blob_data)
        {
            const size_t channels = 3;
            const size_t img_h = img.rows;
            const size_t img_w = img.cols;
            const size_t img_hw = img_h * img_w;
            float *blob_data_ch0 = blob_data;
            float *blob_data_ch1 = blob_data + img_hw;
            float *blob_data_ch2 = blob_data + img_hw * 2;
            // HWC -> CHW
            if (this->model_version_ == "0.1.0")
            {
                for (size_t i = 0; i < img_hw; ++i)
                {
                    // blob = (img / 255.0 - mean) / std
                    const size_t src_idx = i * channels;
                    blob_data_ch0[i] = static_cast<float>(img.data[src_idx + 0]) * this->std255_inv_[0] + this->mean_std_[0];
                    blob_data_ch1[i] = static_cast<float>(img.data[src_idx + 1]) * this->std255_inv_[1] + this->mean_std_[1];
                    blob_data_ch2[i] = static_cast<float>(img.data[src_idx + 2]) * this->std255_inv_[2] + this->mean_std_[2];
                }
            }
            else
            {
                for (size_t i = 0; i < img_hw; ++i)
                {
                    // HWC -> CHW
                    const size_t src_idx = i * channels;
                    blob_data_ch0[i] = static_cast<float>(img.data[src_idx + 0]);
                    blob_data_ch1[i] = static_cast<float>(img.data[src_idx + 1]);
                    blob_data_ch2[i] = static_cast<float>(img.data[src_idx + 2]);
                }
            }

        }
        void blobFromImage_nhwc_cpu(const cv::Mat &img, float *blob_data)
        {
            const size_t channels = 3;
            const size_t img_h = img.rows;
            const size_t img_w = img.cols;
            if (this->model_version_ == "0.1.0")
            {
                for (size_t i = 0; i < img_h * img_w; ++i)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        // blob = (img / 255.0 - mean) / std
                        blob_data[i * channels + c] =
                            static_cast<float>(img.data[i * channels + c]) * this->std255_inv_[c] + this->mean_std_[c];
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < img_h * img_w; ++i)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        blob_data[i * channels + c] = static_cast<float>(img.data[i * channels + c]); // 0.1.1rc0 or later
                    }
                }
            }
        }
    };
}
#endif
