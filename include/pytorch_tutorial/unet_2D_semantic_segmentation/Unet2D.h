#ifndef UNET2D_H
#define UNET2D_H

#include <torch/torch.h>
#include <pytorch_tutorial/unet_2D_semantic_segmentation/double_conv.h>
#include <Eigen/Eigen>

using namespace torch::indexing;

class Unet2DImpl
    : public torch::nn::Module
{
private:
    Eigen::VectorXi features_;
    torch::nn::ModuleList encoder_;
    torch::nn::ModuleList decoder_;
    std::shared_ptr<DoubleConv> neck_layer_;
    std::shared_ptr<torch::nn::Conv2d> output_layer_;
    std::shared_ptr<torch::nn::MaxPool2d> max_pool_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Unet2DImpl(const int64_t _input_channels, const int64_t _output_channels, const std::vector<int64_t> &_features);
    ~Unet2DImpl();
    torch::Tensor forward(torch::Tensor _x);
};

Unet2DImpl::Unet2DImpl(const int64_t _input_channels, const int64_t _output_channels, const std::vector<int64_t> &_features)
{
    /*Unet features*/
    features_ = Eigen::Map<const Eigen::Matrix<int64_t, -1, 1>>(_features.data(), _features.size()).cast<int>();

    /*max pool*/
    max_pool_ = std::make_shared<torch::nn::MaxPool2d>(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

    /*Encoder*/
    auto in_channel = _input_channels;
    auto x = features_.data();
    while (x != features_.data() + features_.size())
    {
        encoder_->push_back(DoubleConv(in_channel, *x));
        in_channel = *x;
        x++;
    }
    register_module("encoder_", this->encoder_);

    /*Decoder*/
    Eigen::VectorXi reversed_features = features_.reverse();
    x = reversed_features.data();
    while (x != reversed_features.data() + reversed_features.size())
    {
        decoder_->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(2 * *x, *x, 2).stride(2)));
        decoder_->push_back(DoubleConv(2 * *x, *x));
        x++;
    }
    std::cout << std::endl;
    register_module("decoder_", decoder_);

    /*Neck layer*/
    neck_layer_ = std::make_shared<DoubleConv>(reversed_features(0), 2 * reversed_features(0));
    register_module("neck_layer_", *neck_layer_);

    /*Output Layer*/
    output_layer_ = std::make_shared<torch::nn::Conv2d>(torch::nn::Conv2d(torch::nn::Conv2dOptions(features_(0), 1, 1)));
    register_module("output_layer_", *output_layer_);
}

Unet2DImpl::~Unet2DImpl()
{
}

torch::Tensor Unet2DImpl::forward(torch::Tensor _x)
{
    if(!_x.requires_grad())
        _x.requires_grad_(true);
    std::vector<torch::Tensor> encoded_features;
    encoded_features.reserve(encoder_->size());

    /*Encoder phase*/
    int i = 0;
    for (auto seq : *encoder_)
    {
        _x = seq->as<DoubleConv>()->forward(_x);
        encoded_features.push_back(_x.clone());
        _x = max_pool_->operator()(_x);
        i++;
    }

    /*Neck phase*/
    _x = neck_layer_->operator()(_x);

    /*Decoder phase*/
    Eigen::Map<const Eigen::Matrix<torch::Tensor, -1, 1>> encoded_features_map(encoded_features.data(), encoded_features.size());
    for (int i = 0; i < decoder_->size(); i += 2)
    {
        _x = decoder_->operator[](i)->as<torch::nn::ConvTranspose2d>()->forward(_x);
        torch::Tensor cat_encoded_features;
        if (_x.sizes() != encoded_features_map.reverse()(std::floor(i / 2)).sizes())
        {
            cat_encoded_features = torch::cat({encoded_features_map.reverse()(std::floor(i / 2)).index({"...",Slice(None,_x.size(-2)),Slice(None,_x.size(-1))}), _x}, -3);
        }else{
            cat_encoded_features = torch::cat({encoded_features_map.reverse()(std::floor(i / 2)), _x}, -3);
        }
        _x = decoder_->operator[](i + 1)->as<DoubleConv>()->forward(cat_encoded_features);
    }

    /*Output phase*/
    _x = output_layer_->operator()(_x);

    return _x;
}

TORCH_MODULE(Unet2D);

#endif