#include "TRTModule.hpp"
#include "ThreadPool.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <numeric>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>

std::string getTypeName(int tag_id) {
    switch (tag_id) {
        case 0: return "类型1";
        case 1: return "类型2";
        default: return "未知";
    }
}

struct FrameResult {
    int sequence;
    cv::Mat frame;
};

std::atomic<bool> exitDisplayThread(false);  // 控制显示线程退出的全局变量

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
    if (choice < 1 or choice > choices.size()) {
        std::cerr << "无效选择" << std::endl;
        return -1;
    }

    std::string selected_model_base_path = choices[choice - 1];
    std::string onnx_path = selected_model_base_path + ".onnx";
    std::string cache_path = selected_model_base_path + ".cache";
    TRTModule trtModule(onnx_path, cache_path);

    ThreadPool pool(std::thread::hardware_concurrency());
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件: " << videoPath << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::vector<double> processingTimes;

    auto compare = [](const FrameResult& lhs, const FrameResult& rhs) {
        return lhs.sequence > rhs.sequence;
    };
    std::priority_queue<FrameResult, std::vector<FrameResult>, decltype(compare)> resultsQueue(compare);
    std::mutex queueMutex;
    std::condition_variable cv;
    std::atomic<int> sequenceNumber = 0;
    std::atomic<int> nextFrameToShow = 0;

    // 启动显示帧的线程
       std::thread display_thread([&]() {
        while (!exitDisplayThread) {
            std::unique_lock<std::mutex> lk(queueMutex);
            cv.wait(lk, [&resultsQueue, &nextFrameToShow] {
                return exitDisplayThread || (!resultsQueue.empty() && resultsQueue.top().sequence <= nextFrameToShow);
            });

            if (exitDisplayThread) {
                break;
            }

            while (!resultsQueue.empty() && resultsQueue.top().sequence <= nextFrameToShow) {
                FrameResult result = resultsQueue.top();
                resultsQueue.pop();
                lk.unlock();

                // 计算平均推理时间
                double avgTime = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0) / processingTimes.size();
                std::string avgTimeLabel = "Average Inference Time: " + std::to_string(avgTime * 1000) + " ms";

                // 在帧上显示平均推理时间
                cv::putText(result.frame, avgTimeLabel, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

                // 显示处理后的帧
                std::cout << "Displaying frame: " << result.sequence << std::endl;
                cv::imshow("Detected Objects", result.frame);
                if (cv::waitKey(30) >= 0) {
                    exitDisplayThread = true;
                    break;
                }

                nextFrameToShow++;
                lk.lock();
            }
        }
    });


    while (cap.read(frame)) {
        cv::Mat frameCopy = frame.clone();
        int currentSequence = sequenceNumber++;

        pool.enqueue([&trtModule, &processingTimes, &queueMutex, &cv, frameCopy, currentSequence, &resultsQueue]() mutable {
            std::cout << "Processing frame: " << currentSequence << std::endl;
            int64 start = cv::getTickCount();

            // 在线程池中处理帧
            std::vector<bbox_t> boxes = trtModule(frameCopy);

            // 在 frameCopy 上绘制 bounding boxes 和标签
            for (const auto& box : boxes) {
                for (int i = 0; i < 4; i++) {
                    cv::line(frameCopy, box.pts[i], box.pts[(i+1)%4], cv::Scalar(0, 255, 0), 2);
                }
                std::string type_name = getTypeName(box.tag_id);
                std::string label = type_name + " - " + std::to_string(box.confidence);
                cv::putText(frameCopy, label, box.pts[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }

            double elapsedTime = (cv::getTickCount() - start) / cv::getTickFrequency();
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                processingTimes.push_back(elapsedTime);
                resultsQueue.push({currentSequence, frameCopy});
                std::cout << "Frame processed: " << currentSequence << std::endl;
            }
            cv.notify_one();
        });
    }

    // 设置退出标志并等待显示线程结束
    exitDisplayThread = true;
    display_thread.join();

    return 0;
}
