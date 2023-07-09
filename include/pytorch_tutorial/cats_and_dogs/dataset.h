#ifndef CAT_AND_DOGS_H
#define CAT_AND_DOGS_H

#include <iostream>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::experimental::filesystem;

// constexpr int kTrainSize = 8007;
constexpr int kTrainSize = 12;
// constexpr int kTestSize = 2027;
constexpr int kTestSize = 8;
constexpr int kRows = 300;
constexpr int kCols = 300;

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &_root, bool _train);

class CatsDogs : public torch::data::datasets::Dataset<CatsDogs>
{
public:
    enum Mode
    {
        kTrain,
        kTest
    };
    explicit CatsDogs(const std::string &_root, Mode _mode = Mode::kTrain);
    torch::data::Example<> get(size_t _index) override;
    torch::optional<size_t> size() const override;
    bool is_train() const noexcept;
    const torch::Tensor &images() const;
    const torch::Tensor &targets() const;
    ~CatsDogs();

private:
    torch::Tensor images_;
    torch::Tensor targets_;
    Mode mode_;
};

CatsDogs::CatsDogs(const std::string &_root, Mode _mode)
    : mode_(_mode)
{
    auto data = read_data(_root, mode_ == CatsDogs::Mode::kTrain);
    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

CatsDogs::~CatsDogs()
{
}

torch::Tensor CVToTensor(cv::Mat _img)
{
    cv::resize(_img, _img, cv::Size(kRows, kCols), 0.0, 0.0, cv::INTER_LINEAR);
    cv::cvtColor(_img, _img, cv::COLOR_BGR2RGB);
    auto img_tensor = torch::from_blob(_img.data, {kRows, kCols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1}).toType(torch::kFloat).div_(255.f);
    return img_tensor;
}

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &_root, bool _train)
{
    int i = 0;
    std::string ext(".jpg");
    const auto num_samples = _train ? kTrainSize : kTestSize;
    const auto folder = _train ? _root + "/train" : _root + "/test";
    auto targets = torch::empty(num_samples, torch::kInt64);
    auto images = torch::empty({num_samples, 3, kRows, kCols}, torch::kFloat);
    std::string cat_folder = folder + "/cats";
    std::string dog_folder = folder + "/dogs";
    std::vector<std::string> folders = {cat_folder, dog_folder};

    for (auto &f : folders)
    {
        int64_t label = 0;
        for (const auto &p : fs::directory_iterator(f))
        {
            if (p.path().extension() == ext)
            {
                cv::Mat img = cv::imread(p.path().string());
                auto img_tensor = CVToTensor(img);
                images[i] = img_tensor;
                targets[i] = torch::tensor(label, torch::kInt64);
            }
            if (i >= num_samples - 1)
                break;
            i++;
        }
        label++;
        i = 0;
    }
    return {images, targets};
}

torch::data::Example<> CatsDogs::get(size_t _index)
{
    return {images_[_index], targets_[_index]};
}

torch::optional<size_t> CatsDogs::size() const
{
    return images_.size(0);
}

bool CatsDogs::is_train() const noexcept
{
    return mode_ == Mode::kTrain;
}

const torch::Tensor &CatsDogs::images() const
{
    return images_;
}

const torch::Tensor &CatsDogs::targets() const
{
    return targets_;
}

#endif