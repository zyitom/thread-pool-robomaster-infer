#include "TRTModule.hpp"
#include "ThreadPool.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <numeric>
#include <vector>

std::string getTypeName(int tag_id) {
    // 根据 tag_id 返回相应的类型名称
    switch (tag_id) {
        case 0: return "类型1";
        case 1: return "类型2";
        // 其他类型...
        default: return "未知";
    }
}

int main() {
    std::string directory = "../models/";
    std::string videoPath = "../../test1.avi";
    std::set<std::string> model_files;

    // 搜索模型文件
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::string path = entry.path().string();
        if (path.ends_with(".onnx") || path.ends_with(".cache")) {
            std::string base_name = path.substr(0, path.find_last_of("."));
            model_files.insert(base_name);
        }
    }

    std::vector<std::string> choices;
    int idx = 0;
    for (const auto& file : model_files) {
        std::cout << ++idx << ". " << file << std::endl;
        choices.push_back(file);
    }

    int choice;
    std::cout << "请选择一个模型（输入数字）: ";
    std::cin >> choice;
    if (choice < 1 || choice > choices.size()) {
        std::cerr << "无效选择" << std::endl;
        return -1;
    }

    std::string selected_model_base_path = choices[choice - 1];
    std::string onnx_path = selected_model_base_path + ".onnx";
    std::string cache_path = selected_model_base_path + ".cache";
    TRTModule trtModule(onnx_path, cache_path);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件: " << videoPath << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::vector<double> processingTimes;

    while (cap.read(frame)) {
        int64 start = cv::getTickCount();

        // 直接在主线程中处理每一帧
        std::vector<bbox_t> boxes = trtModule(frame);

        double elapsedTime = (cv::getTickCount() - start) / cv::getTickFrequency();
        processingTimes.push_back(elapsedTime);

        for (const auto& box : boxes) {
            for (int i = 0; i < 4; i++) {
                cv::line(frame, box.pts[i], box.pts[(i+1)%4], cv::Scalar(0, 255, 0), 2);
            }
            std::string type_name = getTypeName(box.tag_id);
            std::string label = type_name + " - " + std::to_string(box.confidence);
            cv::putText(frame, label, box.pts[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }

        if (!processingTimes.empty()) {
            double avgTimeInMilliseconds = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0) / processingTimes.size() * 1000.0;
            std::string avgTimeLabel = "Avg. Time: " + std::to_string(avgTimeInMilliseconds) + " ms";
            cv::putText(frame, avgTimeLabel, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Detected Objects", frame);
        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    if (!processingTimes.empty()) {
        double totalElapsedTime = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0);
        double avgTimeInMilliseconds = totalElapsedTime / processingTimes.size() * 1000.0;
        std::cout << "Average Processing Time for Entire Video: " << avgTimeInMilliseconds << " ms" << std::endl;
    }

    return 0;
}
