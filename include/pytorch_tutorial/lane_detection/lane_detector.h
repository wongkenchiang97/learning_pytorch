#ifndef LANE_DETECTOR_H
#define LANE_DETECTOR_H

#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <torch/script.h>
#include <torch/torch.h>

class LaneDetector
{
private:
    int64_t default_img_width_;
    int64_t default_img_height_;
    int64_t img_size_;
    cv::VideoCapture cap_;
    torch::jit::Module module_;
    torch::Device device_;
    std::string video_dir_;
    std::string model_dir_;

public:
    LaneDetector(/* args */);
    ~LaneDetector();
    cv::Mat framePrediction(cv::Mat _frame, torch::jit::Module _module);
    torch::jit::Module loadModel(const std::string model_name);
    void streamImage();
};

LaneDetector::LaneDetector(/* args */)
    : device_((torch::cuda::is_available() ? torch::kCUDA : torch::kCPU))
{
    default_img_width_ = 1280;
    default_img_height_ = 720;
    img_size_ = 512;

    /*opencv load images*/
    video_dir_ = "/home/dylan/Downloads/driving.mp4";
    cap_.open(video_dir_);
    if (!cap_.isOpened())
    {
        std::cerr << "failed to open video." << std::endl;
    }

    /*load model*/
    model_dir_ = "/home/dylan/Downloads/quantized_lanesNet.pt";
    try
    {
        module_ = this->loadModel(model_dir_);
    }
    catch (const c10::Error _error)
    {
        std::cerr << "Error: failed to load model." << std::endl;
    }

    std::cout << "press spacebar to terminate." << std::endl;
    streamImage();
}

LaneDetector::~LaneDetector()
{
}

torch::jit::Module LaneDetector::loadModel(const std::string model_name)
{
    torch::jit::Module module = torch::jit::load(model_name);
    module.to(device_);
    module.eval();
    std::cout << "model loaded." << std::endl;
    return module;
}

cv::Mat LaneDetector::framePrediction(cv::Mat _frame, torch::jit::Module _module)
{
    /*overlay opacity*/
    double alpha = 0.25;
    double beta = 1 - alpha;
    cv::Mat frame_copy, dst;

    /*torch model input*/
    std::vector<torch::jit::IValue> input;

    /*mean and stddev*/
    std::vector<double> mean = {0.406, 0.456, 0.485};
    std::vector<double> stddev = {0.225, 0.224, 0.229};
    cv::resize(_frame, _frame, cv::Size(img_size_, img_size_));
    frame_copy = _frame;
    _frame.convertTo(_frame, CV_32FC3, 1.f / 255.f);

    /*cv2 to tensor*/
    torch::Tensor frame_tensor = torch::from_blob(_frame.data, {1, img_size_, img_size_, 3});
    frame_tensor = frame_tensor.permute({0, 3, 1, 2});
    frame_tensor = torch::data::transforms::Normalize<>(mean, stddev)(frame_tensor);
    frame_tensor.to(device_);
    input.push_back(frame_tensor);

    /*model forward path*/
    auto pred = module_.forward(input).toTensor().detach().to(device_);
    pred = pred.mul(100).clamp(0, 255).to(torch::kU8);

    /*Tensor to CV2*/
    cv::Mat detected_mask(cv::Size(img_size_, img_size_), CV_8UC1, pred.data_ptr());
    cv::cvtColor(detected_mask, detected_mask, cv::COLOR_GRAY2RGB);
    cv::applyColorMap(detected_mask, detected_mask, cv::COLORMAP_TWILIGHT_SHIFTED);

    /*overlay prediction mask to color image*/
    cv::addWeighted(frame_copy, alpha, detected_mask, beta, 0.0, dst);
    cv::resize(dst, dst, cv::Size(default_img_width_, default_img_height_));

    return dst;
}

void LaneDetector::streamImage()
{
    cv::Mat img_frame;
    for (;;)
    {
        cap_.read(img_frame);
        if (img_frame.empty())
        {
            std::cerr << "empty frame." << std::endl;
        }

        /*get lane prediction result*/
        img_frame = framePrediction(img_frame, module_);

        cv::imshow("video", img_frame);
        if (cv::waitKey(1) >= 27)
        {
            break;
        }
    }
}

#endif