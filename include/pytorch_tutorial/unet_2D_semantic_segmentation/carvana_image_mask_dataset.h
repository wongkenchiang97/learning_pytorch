#ifndef CARVANA_IMAGE_MASK_DATASET_H
#define CARVANA_IMAGE_MASK_DATASET_H

#include <iostream>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::experimental::filesystem;

constexpr int kTrainSize = 12;
constexpr int kTestSize = 8;
constexpr int kRows = 300;
constexpr int kCols = 300;

namespace carvana_dataset
{

    class ImageMask : public torch::data::Dataset<ImageMask>
    {
    public:
        enum Mode
        {
            kTrain,
            kTest
        };

        explicit ImageMask(const std::string &_root, Mode _mode = Mode::kTrain);
        static std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &_root, bool _train);
        static cv::Mat readMask(const std::string &_path);
        static torch::Tensor CVColorToTensor(cv::Mat _img);
        static torch::Tensor CVMaskToTensor(cv::Mat _img);
        static cv::Mat TensorToCVColor(torch::Tensor _img_tensor);
        static cv::Mat TensorToCVMask(torch::Tensor _img_tensor);

        torch::data::Example<> get(size_t _index) override;
        torch::optional<size_t> size() const override;
        bool is_train() const noexcept;
        const torch::Tensor &images() const;
        const torch::Tensor &targets() const;
        template <typename Loader>
        void showLoaderData(Loader _loader);
        ~ImageMask();

    private:
        torch::Tensor images_;
        torch::Tensor targets_;
        Mode mode_;
    };

    ImageMask::ImageMask(const std::string &_root, Mode _mode)
        : mode_(_mode)
    {
        auto data = read_data(_root, mode_ == ImageMask::Mode::kTrain);
        images_ = std::move(data.first);
        targets_ = std::move(data.second);
    }

    ImageMask::~ImageMask()
    {
    }

    torch::Tensor ImageMask::CVColorToTensor(cv::Mat _img)
    {
        cv::resize(_img, _img, cv::Size(kRows, kCols), 0.0, 0.0, cv::INTER_LINEAR);
        cv::cvtColor(_img, _img, cv::COLOR_BGR2RGB);
        auto img_tensor = torch::from_blob(_img.data, {kRows, kCols, 3}, torch::kByte);
        img_tensor = img_tensor.permute({2, 0, 1}).toType(torch::kFloat).div_(255.f);
        return img_tensor;
    }

    torch::Tensor ImageMask::CVMaskToTensor(cv::Mat _img)
    {
        cv::resize(_img, _img, cv::Size(kRows, kCols), 0.0, 0.0, cv::INTER_LINEAR);
        cv::cvtColor(_img, _img, cv::COLOR_RGB2GRAY);
        auto img_tensor = torch::from_blob(_img.data, {kRows, kCols}, torch::kByte);
        img_tensor = img_tensor.permute({0, 1}).toType(torch::kFloat).div_(255.f);
        return img_tensor;
    }

    cv::Mat ImageMask::TensorToCVColor(torch::Tensor _img_tensor)
    {
        _img_tensor = _img_tensor.permute({1, 2, 0});
        _img_tensor = _img_tensor.mul(0.5).add(0.5).mul(255).clamp(0, 255).to(torch::kByte);
        _img_tensor = _img_tensor.contiguous();

        int height = _img_tensor.size(0);
        int width = _img_tensor.size(1);
        cv::Mat output(cv::Size(width, height), CV_8UC3);
        std::memcpy((void *)output.data, _img_tensor.data_ptr(), sizeof(torch::kU8) * _img_tensor.numel());
        cv::cvtColor(output, output, cv::COLOR_BGR2RGB);

        return output.clone();
    }
    cv::Mat ImageMask::TensorToCVMask(torch::Tensor _img_tensor)
    {
        _img_tensor = _img_tensor.permute({0, 1});
        _img_tensor = _img_tensor.mul(0.5).add(0.5).mul(255).clamp(0, 255).to(torch::kByte);
        _img_tensor = _img_tensor.contiguous();

        int height = _img_tensor.size(0);
        int width = _img_tensor.size(1);
        cv::Mat output(cv::Size(width, height), CV_8UC1);
        std::memcpy((void *)output.data, _img_tensor.data_ptr(), sizeof(torch::kU8) * _img_tensor.numel());

        return output.clone();
    }

    cv::Mat ImageMask::readMask(const std::string &_path)
    {
        cv::VideoCapture cap_;
        cap_.open(_path);
        if (!cap_.isOpened())
        {
            std::cerr << "failed to open video." << std::endl;
        }
        cv::Mat img_frame;
        cap_.read(img_frame);
        if (img_frame.empty())
        {
            std::cerr << "empty frame." << std::endl;
        }
        cap_.release();
        // cv::resize(img_frame, img_frame, cv::Size(640, 480), 0.0, 0.0, cv::INTER_LINEAR);
        return img_frame.clone();
    }

    std::pair<torch::Tensor, torch::Tensor> ImageMask::read_data(const std::string &_root, bool _train)
    {
        int i = 0;
        std::vector<std::string> extensions = {".jpg", ".gif"};
        const auto num_samples = _train ? kTrainSize : kTestSize;
        const auto folder = _train ? _root + "/train" : _root + "/val";
        auto targets = torch::empty({num_samples, kRows, kCols}, torch::kFloat);
        auto images = torch::empty({num_samples, 3, kRows, kCols}, torch::kFloat);
        std::string img_folder = folder + "/images";
        std::string mask_folder = folder + "/masks";
        std::vector<std::string> folders = {img_folder, mask_folder};
        std::vector<std::string> mask_filenames;

        /*read color image*/
        for (const auto &p : fs::directory_iterator(folders[0]))
        {
            if (p.path().extension() != extensions[0])
            {
                continue;
            }
            mask_filenames.push_back(p.path().stem().string() + "_mask");
            cv::Mat img = cv::imread(p.path().string());
            auto img_tensor = CVColorToTensor(img);
            images[i] = img_tensor;
            if (i >= num_samples - 1)
                break;
            i++;
        }
        i = 0;

        /*read mask image*/
        for (const auto &p : fs::directory_iterator(folders[1]))
        {
            if (p.path().extension() != extensions[1])
            {
                continue;
            }

            cv::Mat mask = readMask(p.path().parent_path().append(mask_filenames[i] + extensions[1]).string());
            targets[i] = CVMaskToTensor(mask);

            if (i >= num_samples - 1)
                break;
            i++;
        }

        return {images, targets};
    }

    torch::data::Example<> ImageMask::get(size_t _index)
    {
        return {images_[_index], targets_[_index]};
    }

    torch::optional<size_t> ImageMask::size() const
    {
        return images_.size(0);
    }

    bool ImageMask::is_train() const noexcept
    {
        return mode_ == Mode::kTrain;
    }

    const torch::Tensor &ImageMask::images() const
    {
        return images_;
    }

    const torch::Tensor &ImageMask::targets() const
    {
        return targets_;
    }

    template <typename Loader>
    void ImageMask::showLoaderData(Loader _loader)
    {
        int64_t batch_idx = 0;
        for (auto batch : *_loader)
        {
            std::cout << "batch[" << batch_idx << "]" << std::endl;
            for (int i = 0; i < batch.data.sizes().at(0); i++)
            {
                auto tensor_img = batch.data[i];
                auto tensor_mask = batch.target[i];

                auto img = carvana_dataset::ImageMask::TensorToCVColor(tensor_img);
                auto mask = carvana_dataset::ImageMask::TensorToCVMask(tensor_mask);

                cv::Mat dst;
                double alpha = 0.5;
                double beta = 1 - alpha;
                cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
                cv::applyColorMap(mask, mask, cv::COLORMAP_TWILIGHT_SHIFTED);
                cv::addWeighted(img, alpha, mask, beta, 0.0, dst);
                cv::hconcat(dst, mask, dst);
                imshow("loader data", dst);
                cv::waitKey(3e3);
            }
            batch_idx++;
        }
    }

} // namespace carvana_dataset

#endif