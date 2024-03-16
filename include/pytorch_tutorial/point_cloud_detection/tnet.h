#ifndef TNET_H
#define TNET_H

#include <torch/nn/functional.h>
#include <torch/torch.h>

class TNetImpl : public torch::nn::Module {
public:
    TNetImpl(int64_t _spatial_dim, int64_t _num_pts);
    ~TNetImpl();
    torch::Tensor forward(torch::Tensor _x);

private:
    int64_t spatial_dim_;

    torch::nn::ModuleList conv1d_, linear_, batch_norm_;
    std::shared_ptr<torch::nn::MaxPool1d> max_pool_;
};

TNetImpl::TNetImpl(int64_t _spatial_dim, int64_t _num_pts)
    : spatial_dim_(_spatial_dim)
{
    /*Conv1d*/
    conv1d_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(spatial_dim_, 64, 1)));
    conv1d_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 1)));
    conv1d_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 1024, 1)));
    register_module("conv1d", conv1d_);

    /*Linear*/
    linear_->push_back(torch::nn::Linear(torch::nn::LinearOptions(1024, 512)));
    linear_->push_back(torch::nn::Linear(torch::nn::LinearOptions(512, 256)));
    linear_->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, std::pow(_spatial_dim, 2))));
    register_module("linear", linear_);

    /*BatchNorm*/
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(64)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(128)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1024)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(256)));
    register_module("batch_norm", batch_norm_);

    /*Max Pool*/
    max_pool_ = std::make_shared<torch::nn::MaxPool1d>(torch::nn::MaxPool1dOptions(_num_pts));
}

TNetImpl::~TNetImpl()
{
}

torch::Tensor TNetImpl::forward(torch::Tensor _x)
{
    auto batch_size = _x.size(0);

    /*Shared MLP(Conv1d) Layers*/
    auto it_conv = conv1d_->begin();
    auto it_bn = batch_norm_->begin();
    while (it_conv != conv1d_->end()) {
        size_t idx = std::distance(conv1d_->begin(), it_conv);
        _x = (*it_conv)->as<torch::nn::Conv1d>()->forward(_x); // conv1d
        _x = torch::relu(_x); // relu
        _x = (*it_bn)->as<torch::nn::BatchNorm1d>()->forward(_x); // batch norm
        it_conv++;
        it_bn++;
    }

    /*Max Pool over num of pts*/
    _x = max_pool_->operator()(_x).view({ batch_size, -1 });

    /*MLP(Linear) Layers*/
    auto it_linear = linear_->begin();
    while (it_linear != linear_->end()-1) {
        _x = (*it_linear)->as<torch::nn::Linear>()->forward(_x); // linear
        _x = torch::relu(_x); // relu
        _x = (*it_bn)->as<torch::nn::BatchNorm1d>()->forward(_x); // batch norm

        it_linear++;
        it_bn++;
    }
    _x = (*it_linear)->as<torch::nn::Linear>()->forward(_x); // linear

    /*Set Identity Tensors*/
    auto device = (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    auto identity = torch::eye(spatial_dim_, torch::TensorOptions().requires_grad(true).device(device)).repeat({ batch_size, 1, 1 });

    /*Set Canonical Transform*/
    _x = _x.view({ -1, spatial_dim_, spatial_dim_ }) + identity;

    return _x;
}

TORCH_MODULE(TNet);

#endif