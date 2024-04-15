#ifndef POINTNET_H
#define POINTNET_H

#include <pytorch_tutorial/point_cloud_detection/tnet.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

class PointNetBackboneImpl
    : public torch::nn::Module {
public:
    PointNetBackboneImpl(int64_t _num_pts, int64_t _num_global_feature, bool _local_feature);
    ~PointNetBackboneImpl();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor _x);

private:
    int64_t num_pts_, num_global_feature_, num_class_;
    bool local_feature_;
    torch::nn::ModuleList spatial_transformers_, mlp1_, mlp2_, batch_norm_;
};

PointNetBackboneImpl::PointNetBackboneImpl(int64_t _num_pts, int64_t _num_global_feature, bool _local_feature)
    : num_pts_(_num_pts)
    , num_global_feature_(_num_global_feature)
    , local_feature_(_local_feature)
{
    /*Spatial Transformers*/
    spatial_transformers_->push_back(TNet(3, _num_pts));
    spatial_transformers_->push_back(TNet(64, _num_pts));
    register_module("spatial_transformers", spatial_transformers_);

    /*MLP1(64,64)*/
    mlp1_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 64, 1)));
    mlp1_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 64, 1)));
    register_module("mlp1", mlp1_);

    /*MLP2(64,128,_num_global_feature)*/
    mlp2_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 64, 1)));
    mlp2_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 1)));
    mlp2_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(128, _num_global_feature, 1)));
    register_module("mlp2", mlp2_);

    /*BatchNorm*/
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(64)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(64)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(64)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(128)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(_num_global_feature)));
    register_module("batch_norm", batch_norm_);
}

PointNetBackboneImpl::~PointNetBackboneImpl()
{
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PointNetBackboneImpl::forward(torch::Tensor _x)
{
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backbone_feature; //<features,critical_pts_idx,feature_transform>
    auto batch_size = _x.size(0);

    /*Spatial Transform(3x3)*/
    auto A_input = spatial_transformers_->at<torch::nn::Module>(0).as<TNet>()->forward(_x);
    _x = torch::bmm(_x.transpose(2, 1), A_input).transpose(2, 1);

    /*MLP(64,64)*/
    auto it_mlp1 = mlp1_->begin();
    auto it_bn = batch_norm_->begin();
    while (it_mlp1 != mlp1_->end()) {
        _x = (*it_mlp1)->as<torch::nn::Conv1d>()->forward(_x);
        _x = torch::nn::functional::relu(_x);
        _x = (*it_bn)->as<torch::nn::BatchNorm1d>()->forward(_x);
        it_mlp1++;
        it_bn++;
    }

    /*Spatial Transform(64x64)*/
    std::get<2>(backbone_feature) = spatial_transformers_->at<torch::nn::Module>(1).as<TNet>()->forward(_x);
    _x = torch::bmm(_x.transpose(2, 1), std::get<2>(backbone_feature)).transpose(2, 1);
    auto local_feature = _x.clone().to(_x.device());

    /*MLP(64,128,1024)*/
    auto it_mlp2 = mlp2_->begin();
    while (it_mlp2 != mlp2_->end()) {
        _x = (*it_mlp2)->as<torch::nn::Conv1d>()->forward(_x);
        _x = torch::nn::functional::relu(_x);
        _x = (*it_bn)->as<torch::nn::BatchNorm1d>()->forward(_x);
        it_mlp2++;
        it_bn++;
    }

    /*Get Critical point index and Global feature*/
    auto max_pool_opt = torch::nn::functional::MaxPool1dFuncOptions(num_pts_);
    auto pool_result = torch::nn::functional::max_pool1d_with_indices(_x, max_pool_opt);
    std::get<1>(backbone_feature) = std::get<1>(pool_result).view({ batch_size, -1 });

    if (local_feature_) {
        /*Feature:{local,global}*/
        std::get<0>(backbone_feature) = torch::cat({ local_feature, std::get<0>(pool_result).view({ batch_size, -1 }).unsqueeze_(-1).repeat({ 1, 1, num_pts_ }) }, 1);
    } else {
        /*Feature:{global}*/
        std::get<0>(backbone_feature) = std::get<0>(pool_result).view({ batch_size, -1 });
    }

    return backbone_feature;
}

TORCH_MODULE(PointNetBackbone);

class PointNetClassificationImpl
    : public torch::nn::Module {
public:
    PointNetClassificationImpl(int64_t _num_pts, int64_t _num_global_feature, int64_t _num_class);
    ~PointNetClassificationImpl();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor _x);

private:
    int64_t num_pts_, num_global_feature_, num_class_;
    PointNetBackbone backbone_ { nullptr };
    torch::nn::ModuleList mlp_, batch_norm_;
    torch::nn::Dropout dropout_ { nullptr };
};

PointNetClassificationImpl::PointNetClassificationImpl(int64_t _num_pts, int64_t _num_global_feature, int64_t _num_class)
    : num_pts_(_num_pts)
    , num_global_feature_(_num_global_feature)
    , num_class_(_num_class)
{
    /*Set Network Backbone*/
    backbone_ = PointNetBackbone(_num_pts, _num_global_feature, false);
    register_module("backbone", backbone_);

    /*MLP(512,256,_num_class)*/
    mlp_->push_back(torch::nn::Linear(torch::nn::LinearOptions(_num_global_feature, 512)));
    mlp_->push_back(torch::nn::Linear(torch::nn::LinearOptions(512, 256)));
    mlp_->push_back(torch::nn::Linear(torch::nn::LinearOptions(256, _num_class)));
    register_module("mlp", mlp_);

    /*BatchNorm*/
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(256)));
    register_module("batch_norm", batch_norm_);

    /*Dropout*/
    dropout_ = torch::nn::Dropout(torch::nn::DropoutOptions(0.6));
}

PointNetClassificationImpl::~PointNetClassificationImpl()
{
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PointNetClassificationImpl::forward(torch::Tensor _x)
{
    /*Get backbone features*/
    auto backbone_feature = backbone_->forward(_x);

    /*MLP(Classification)*/
    auto it_mlp = mlp_->begin();
    auto it_bn = batch_norm_->begin();
    while (it_mlp != mlp_->end() - 1) {
        std::get<0>(backbone_feature) = (*it_mlp)->as<torch::nn::Linear>()->forward(std::get<0>(backbone_feature));
        std::get<0>(backbone_feature) = torch::nn::functional::relu(std::get<0>(backbone_feature));
        std::get<0>(backbone_feature) = (*it_bn)->as<torch::nn::BatchNorm1d>()->forward(std::get<0>(backbone_feature));
        it_mlp++;
        it_bn++;
    }
    std::get<0>(backbone_feature) = (*it_mlp)->as<torch::nn::Linear>()->forward(std::get<0>(backbone_feature));
    std::get<0>(backbone_feature) = dropout_->forward(std::get<0>(backbone_feature));

    return backbone_feature;
}

TORCH_MODULE(PointNetClassification);

class PointNetSegmentationImpl
    : public torch::nn::Module {
public:
    PointNetSegmentationImpl(int64_t _num_pts, int64_t _num_global_feature, int64_t _num_class);
    ~PointNetSegmentationImpl();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor _x);

private:
    int64_t num_pts_, num_global_feature_, num_class_;
    PointNetBackbone backbone_ { nullptr };
    torch::nn::ModuleList mlp_, batch_norm_;
};

PointNetSegmentationImpl::PointNetSegmentationImpl(int64_t _num_pts, int64_t _num_global_feature, int64_t _num_class)
    : num_pts_(_num_pts)
    , num_global_feature_(_num_global_feature)
    , num_class_(_num_class)
{
    /*Set Network Backbone*/
    backbone_ = PointNetBackbone(_num_pts, _num_global_feature, true);
    register_module("backbone", backbone_);

    /*MLP(Segmentation)*/
    mlp_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(64 + _num_global_feature, 512, 1)));
    mlp_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(512, 256, 1)));
    mlp_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(256, 128, 1)));
    mlp_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(128, _num_class, 1)));
    register_module("mlp", mlp_);

    /*BatchNorm*/
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(256)));
    batch_norm_->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(128)));
    register_module("batch_norm", batch_norm_);
}

PointNetSegmentationImpl::~PointNetSegmentationImpl()
{
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PointNetSegmentationImpl::forward(torch::Tensor _x)
{
    /*Get backbone features*/
    auto backbone_feature = backbone_->forward(_x);

    /*MLP(Segmentation)*/
    auto it_mlp = mlp_->begin();
    auto it_bn = batch_norm_->begin();
    while (it_mlp != mlp_->end() - 1) {
        std::get<0>(backbone_feature) = (*it_mlp)->as<torch::nn::Conv1d>()->forward(std::get<0>(backbone_feature));
        std::get<0>(backbone_feature) = torch::nn::functional::relu(std::get<0>(backbone_feature));
        std::get<0>(backbone_feature) = (*it_bn)->as<torch::nn::BatchNorm1d>()->forward(std::get<0>(backbone_feature));
        it_mlp++;
        it_bn++;
    }
    std::get<0>(backbone_feature) = (*it_mlp)->as<torch::nn::Conv1d>()->forward(std::get<0>(backbone_feature));
    std::get<0>(backbone_feature).transpose_(2, 1);

    return backbone_feature;
}

TORCH_MODULE(PointNetSegmentation);

#endif