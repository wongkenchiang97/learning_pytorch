#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>

class ModelImpl
    : public torch::nn::Module
{
private:
    torch::nn::Linear fc1_, fc2_;

public:
    ModelImpl(int64_t _input_size, int64_t _hidden_size, int64_t _num_class);
    ~ModelImpl();
    torch::Tensor forward(torch::Tensor _x);
};

ModelImpl::ModelImpl(int64_t _input_size, int64_t _hidden_size, int64_t _num_class)
    : fc1_(_input_size, _hidden_size),
      fc2_(_hidden_size, _num_class)
{
    register_module("fc1_", fc1_);
    register_module("fc2_", fc2_);
}

ModelImpl::~ModelImpl()
{
}

torch::Tensor ModelImpl::forward(torch::Tensor _x)
{
    _x = torch::nn::functional::relu(fc1_->forward(_x));
    return fc2_->forward(_x);
}

TORCH_MODULE(Model);

#endif