#ifndef DOUBLE_CONV_H
#define DOUBLE_CONV_H

#include <torch/torch.h>

class DoubleConvImpl
    : public torch::nn::Module
{
private:
    torch::nn::Sequential double_conv{nullptr};

public:
    DoubleConvImpl(int64_t _in_channels, int64_t _out_channels);
    ~DoubleConvImpl();
    torch::Tensor forward(torch::Tensor _x);
};

DoubleConvImpl::DoubleConvImpl(int64_t _in_channels, int64_t _out_channels)
{
    double_conv = torch::nn::Sequential();
    double_conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(_in_channels, _out_channels, 3).stride(1).padding(1).bias(false)));
    double_conv->push_back(torch::nn::BatchNorm2d(_out_channels));
    double_conv->push_back(torch::nn::ReLU());
    double_conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(_out_channels, _out_channels, 3).stride(1).padding(1).bias(false)));
    double_conv->push_back(torch::nn::BatchNorm2d(_out_channels));
    double_conv->push_back(torch::nn::ReLU());

    register_module("double_conv", this->double_conv);
}

DoubleConvImpl::~DoubleConvImpl()
{
}

torch::Tensor DoubleConvImpl::forward(torch::Tensor _x)
{
    return double_conv->forward(_x);
}

TORCH_MODULE(DoubleConv);

#endif