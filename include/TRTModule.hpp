//
// Created by xinyang on 2021/4/8.
//

#ifndef _ONNXTRTMODULE_HPP_
#define _ONNXTRTMODULE_HPP_

#include <opencv2/core.hpp>
#include <NvInfer.h>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

struct alignas(4) bbox_t {
    cv::Point2f pts[4]; // [pt0, pt1, pt2, pt3]
    float confidence;
    int color_id; // 0: blue, 1: red, 2: gray
    int tag_id;   // 0: guard, 1-5: number, 6: base

    bool operator==(const bbox_t& other) const {
        return std::equal(std::begin(pts), std::end(pts), std::begin(other.pts)) &&
               confidence == other.confidence &&
               color_id == other.color_id &&
               tag_id == other.tag_id;
    }

    bool operator!=(const bbox_t& other) const {
        return !(*this == other);
    }
};

/*
 * 四点模型
 */
class TRTModule {
    static constexpr int TOPK_NUM = 128;
    static constexpr float KEEP_THRES = 0.1f;

public:
    explicit TRTModule(const std::string &onnx_file);
    TRTModule(const std::string &modelPath, bool isOnnx);
    TRTModule(const std::string &onnx_file, const std::string &cache_file);

    ~TRTModule();
    std::vector<std::vector<bbox_t>> operator()(const std::vector<cv::Mat> &src_batch) const;
    TRTModule(const TRTModule &) = delete;

    TRTModule operator=(const TRTModule &) = delete;

    std::vector<bbox_t> operator()(const cv::Mat &src) const;

private:
    void build_engine_from_onnx(const std::string &onnx_file);

    void build_engine_from_cache(const std::string &cache_file);

    void cache_engine(const std::string &cache_file);

    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    mutable void *device_buffer[2];
    float *output_buffer;
    cudaStream_t stream;
    int input_idx, output_idx;
    size_t input_sz, output_sz;
};


#endif /* _ONNXTRTMODULE_HPP_ */
