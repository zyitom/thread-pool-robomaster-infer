#include "TRTModule.hpp"
#include "ThreadPool.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <numeric>
#include <vector>
#include <deque>
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

std::atomic<bool> exitDisplayThread(false);
std::mutex queueMutex;
std::condition_variable conditionVar;

int main() {
    std::string directory = "../models/";
    std::string videoPath = "../../test1.avi";
    std::set<std::string> model_files;
    const size_t maxQueueSize = 100;

    // 搜索模型文件
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::string path = entry.path().string();
        if (path.ends_with(".onnx") || path.ends_with(".cache")) {
            std::string base_name = path.substr(0, path.find_last_of("."));
            model_files.insert(base_name);
        }
    }

    std::vector<std::string> choices;
    for (const auto& file : model_files) {
        choices.push_back(file);
        std::cout << choices.size() << ". " << file << std::endl;
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

    ThreadPool pool(std::thread::hardware_concurrency());
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件: " << videoPath << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::vector<double> processingTimes;
    std::deque<FrameResult> frameQueue;
    std::atomic<int> sequenceNumber = 0;
    std::atomic<int> nextFrameToShow = 0;

    std::thread display_thread([&]() {
    while (!exitDisplayThread) {
        std::unique_lock<std::mutex> lk(queueMutex);
        conditionVar.wait(lk, [&frameQueue, &nextFrameToShow] {
            return exitDisplayThread || (!frameQueue.empty() && frameQueue.front().sequence == nextFrameToShow);
        });

        while (!frameQueue.empty() && frameQueue.front().sequence == nextFrameToShow) {
            FrameResult result = frameQueue.front();
            frameQueue.pop_front();

            double avgTime = 0.0;
            if (!processingTimes.empty()) {
                avgTime = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0) / processingTimes.size();
            }
            std::string avgTimeLabel = "Avg Inference Time: " + std::to_string(avgTime * 1000) + " ms";
            cv::putText(result.frame, avgTimeLabel, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            // 在帧上显示已处理的帧数
            std::cout << "已显示 " << nextFrameToShow << " 帧" << std::endl; // 添加这一行来监控已显示的帧数
std::string frameCountLabel = "Processed Frames: " + std::to_string(nextFrameToShow);
cv::putText(result.frame, frameCountLabel, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

                cv::imshow("Detected Objects", result.frame);
                if (cv::waitKey(2) >= 0) {
                    exitDisplayThread = true;
                    break;
                }

                nextFrameToShow++;
            }
            lk.unlock();
            conditionVar.notify_all();
        }
    });

    while (cap.read(frame)) {
        std::unique_lock<std::mutex> lock(queueMutex);
        if (frameQueue.size() >= maxQueueSize) {
            conditionVar.wait(lock, [&frameQueue, &maxQueueSize] {
                return frameQueue.size() < maxQueueSize;
            });
        }
        lock.unlock();
std::cout << "处理第 " << sequenceNumber << " 帧" << std::endl; // 添加这一行来监控正在处理的帧序号
        cv::Mat frameCopy = frame.clone();
        int currentSequence = sequenceNumber++;

        pool.enqueue([&trtModule, &processingTimes, frameCopy, currentSequence, &frameQueue]() mutable {
    int64 start = cv::getTickCount();


            std::vector<bbox_t> boxes = trtModule(frameCopy);

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
        frameQueue.push_back({currentSequence, frameCopy});
        conditionVar.notify_one();
            }
            conditionVar.notify_one();
        });
    }

    exitDisplayThread = true;
    display_thread.join();

    return 0;
}
