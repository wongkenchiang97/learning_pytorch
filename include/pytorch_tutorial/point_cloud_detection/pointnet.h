#ifndef POINTNET_H
#define POINTNET_H

#include <pytorch_tutorial/point_cloud_detection/tnet.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

#define PCL_NO_PRECOMPILE
#include <Eigen/Eigen>
// #include <pcl-1.13/pcl/filters/farthest_point_sampling.h>
// #include <pcl-1.13/pcl/types.h>
#include <pcl/common/colors.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pytorch_tutorial/point_cloud_detection/point_type.h>

using namespace torch::indexing;

namespace pointnet {

template <typename PointT>
torch::Tensor pointCloudToTensor(typename pcl::PointCloud<PointT>::ConstPtr _in)
{
    auto pts_map = Eigen::Map<const Eigen::MatrixXf, 0, Eigen::Stride<sizeof(PointT) / sizeof(float), 1>>(&_in->points[0].x, 3, _in->points.size());
    Eigen::MatrixXf xyz = pts_map;
    Eigen::Vector3f min = xyz.rowwise().minCoeff();
    Eigen::Vector3f max = xyz.rowwise().maxCoeff();
    xyz.row(0).array() = (xyz.row(0).array() - min(0)) / (max(0) - min(0));
    xyz.row(1).array() = (xyz.row(1).array() - min(1)) / (max(1) - min(1));
    xyz.row(2).array() = (xyz.row(2).array() - min(2)) / (max(2) - min(2));
    torch::Tensor tensor_data = torch::from_blob(xyz.data(), { 3, xyz.cols() }, torch::TensorOptions().dtype(torch::kFloat)).clone();
    tensor_data = tensor_data.view({ xyz.cols(), 3 });
    return tensor_data;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr tensorToPointCloud(torch::Tensor _in)
{
    auto xyz_tensor = _in.index({ "...", Slice(0, 3, None) }).to(torch::kCPU).transpose(-2, -1);
    auto xyz_map = Eigen::Map<Eigen::MatrixXf>(xyz_tensor.data_ptr<float>(), xyz_tensor.size(-2), xyz_tensor.size(-1));
    typename pcl::PointCloud<PointT>::Ptr xyz_cloud;
    xyz_cloud = pcl::make_shared<pcl::PointCloud<PointT>>();
    xyz_cloud->points.resize(_in.size(-2));
    xyz_cloud->getMatrixXfMap(3, sizeof(PointT) / sizeof(float), 0) = xyz_map;
    return xyz_cloud;
}

class PointNetBackboneImpl
    : public torch::nn::Module {
public:
    PointNetBackboneImpl(int64_t _num_input_feature, int64_t _num_pts, int64_t _num_global_feature, bool _local_feature = false);
    virtual ~PointNetBackboneImpl();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor _x);

private:
    int64_t num_input_feature_, num_pts_, num_global_feature_, num_class_;
    bool local_feature_;
    torch::nn::ModuleList spatial_transformers_, mlp1_, mlp2_, batch_norm_;
};

PointNetBackboneImpl::PointNetBackboneImpl(int64_t _num_input_feature, int64_t _num_pts, int64_t _num_global_feature, bool _local_feature)
    : num_input_feature_(_num_input_feature)
    , num_pts_(_num_pts)
    , num_global_feature_(_num_global_feature)
    , local_feature_(_local_feature)
{
    /*Spatial Transformers*/
    spatial_transformers_->push_back(TNet(num_input_feature_, _num_pts));
    spatial_transformers_->push_back(TNet(64, _num_pts));
    register_module("spatial_transformers", spatial_transformers_);

    /*MLP1(64,64)*/
    mlp1_->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(num_input_feature_, 64, 1)));
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
    backbone_ = PointNetBackbone(3, _num_pts, _num_global_feature, false);
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
    backbone_ = PointNetBackbone(3, _num_pts, _num_global_feature, true);
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

class SetAbstractionImpl
    : public torch::nn::Module {
public:
    SetAbstractionImpl(int64_t _num_input_feature, int64_t _num_group, float _nn_radius, int64_t _num_pts, int64_t _num_global_feature);
    virtual ~SetAbstractionImpl();
    torch::Tensor forward(torch::Tensor _x);

private:
    pcl::IndicesPtr randomSampling(pcl::PointCloud<pcl::PointXYZ>::ConstPtr _input_xyz, int64_t _num_of_pts);
    std::vector<pcl::Indices> grouping(pcl::PointCloud<pcl::PointXYZ>::ConstPtr _input_xyz, pcl::IndicesPtr _centroid, float _radius, int64_t _max_nn);
    torch::Tensor miniPointnet(torch::Tensor _x, std::vector<pcl::Indices>& _groups_idx);

    int64_t num_group_, num_pts_, num_global_feature_;
    float nn_radius_;
    torch::Device device_;
    PointNetBackbone mini_pointnets_;
};

SetAbstractionImpl::SetAbstractionImpl(int64_t _num_input_feature, int64_t _num_group, float _nn_radius, int64_t _num_pts, int64_t _num_global_feature)
    : num_group_(_num_group)
    , nn_radius_(_nn_radius)
    , num_pts_(_num_pts)
    , num_global_feature_(_num_global_feature)
    , device_((torch::cuda::is_available() ? torch::kCUDA : torch::kCPU))
    , mini_pointnets_(_num_input_feature, num_pts_, num_global_feature_)
{
    mini_pointnets_->to(device_);
}

SetAbstractionImpl::~SetAbstractionImpl()
{
}

torch::Tensor SetAbstractionImpl::forward(torch::Tensor _x)
{
    if (_x.size(-1) < 3)
        std::runtime_error("feature dimension must be at least 3");
    if (num_group_ > _x.size(-2))
        std::runtime_error("number of group has exceed number of samples.");

    torch::Tensor out = torch::empty({ _x.size(0), num_group_, 3 + num_global_feature_ }, torch::TensorOptions().device(_x.device()));
    torch::Tensor grouped_pts = torch::empty({ _x.size(0), num_group_, num_pts_, 3 }, torch::TensorOptions().device(_x.device()));
    for (int i = 0; i < _x.size(0); i++) {
        /*Tensor to PointCloud*/
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_xyz;
        input_xyz = tensorToPointCloud<pcl::PointXYZ>(_x.index({ i, "...", Slice(None, 3, None) }).to(torch::kCPU));

        /*Sampling*/
        auto centroid_idx = randomSampling(input_xyz, num_group_);
        torch::Tensor idx = torch::from_blob(centroid_idx->data(), { centroid_idx->size() }, torch::TensorOptions().dtype(torch::kInt)).clone().to(torch::kLong).to(_x.device());
        out.index_put_({ i, Slice(), Slice(None, 3, None) }, _x.index_select(-2, idx).index({ i, Slice(), Slice(None, 3, None) }));

        /*Grouping*/
        auto groups_idx = grouping(input_xyz, centroid_idx, nn_radius_, num_pts_);
        for (int group = 0; group < num_group_; group++) {
            /*Set Grouped Points*/
            const torch::Tensor group_idx = torch::from_blob(groups_idx[group].data(), { groups_idx[group].size() }, torch::TensorOptions().dtype(torch::kInt)).clone().to(torch::kLong).to(_x.device());
            grouped_pts.index_put_({ i, group, "..." }, _x.index_select(-2, group_idx).index({ i, Slice(), Slice(None, 3, None) }));

            /*Min Max Normalize*/
            grouped_pts.index_put_({ i, group, Slice(), 0 }, (grouped_pts.index({ i, group, Slice(), 0 }).sub(grouped_pts.index({ i, group, Slice(), 0 }).min().item<float>())).div(grouped_pts.index({ i, group, Slice(), 0 }).max().item<float>() - grouped_pts.index({ i, group, Slice(), 0 }).min().item<float>()));
            grouped_pts.index_put_({ i, group, Slice(), 1 }, (grouped_pts.index({ i, group, Slice(), 1 }).sub(grouped_pts.index({ i, group, Slice(), 1 }).min().item<float>())).div(grouped_pts.index({ i, group, Slice(), 1 }).max().item<float>() - grouped_pts.index({ i, group, Slice(), 1 }).min().item<float>()));
            grouped_pts.index_put_({ i, group, Slice(), 2 }, (grouped_pts.index({ i, group, Slice(), 2 }).sub(grouped_pts.index({ i, group, Slice(), 2 }).min().item<float>())).div(grouped_pts.index({ i, group, Slice(), 2 }).max().item<float>() - grouped_pts.index({ i, group, Slice(), 2 }).min().item<float>()));
        }
    }

    /*Mini PointNet*/
    for (int group = 0; group < num_group_; group++) {
        out.index_put_({ "...", group, Slice(3) }, std::get<0>(mini_pointnets_->forward(grouped_pts.index({ "...", group, Slice(), Slice(None, 3, None) }).transpose(-2, -1))));
    }

    return out;
}

pcl::IndicesPtr SetAbstractionImpl::randomSampling(pcl::PointCloud<pcl::PointXYZ>::ConstPtr _input_xyz, int64_t _num_of_sample)
{
    assert(_input_xyz != nullptr && !_input_xyz->empty());

    /*Random Sampling*/
    pcl::IndicesPtr idx = pcl::make_shared<pcl::Indices>();
    typename pcl::RandomSample<pcl::PointXYZ> sample;
    sample.setInputCloud(_input_xyz);
    sample.setSample(_num_of_sample);
    sample.filter(*idx);

    return idx;
}

std::vector<pcl::Indices> SetAbstractionImpl::grouping(pcl::PointCloud<pcl::PointXYZ>::ConstPtr _input_xyz, pcl::IndicesPtr _centroid_idx, float _radius, int64_t _max_pts)
{
    assert(_input_xyz != nullptr && !_input_xyz->empty());
    assert(_centroid_idx != nullptr && !_centroid_idx->empty());
    assert(_centroid_idx->size() == num_group_);
    std::vector<pcl::Indices> groups_idx(_centroid_idx->size());

    /*Radius search*/
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    for (int i = 0; i < _centroid_idx->size(); i++) {
        std::vector<int> indices;
        std::vector<float> distances;
        kdtree.setInputCloud(_input_xyz);
        int nn_num = kdtree.radiusSearch(_input_xyz->points[(*_centroid_idx)[i]], _radius, groups_idx[i], distances, (uint)(_max_pts));
        if (nn_num < _max_pts) {
            /*Centroid idx padding*/
            pcl::Indices centroid_idx_copy(_max_pts - nn_num, (*_centroid_idx)[i]);
            groups_idx[i].insert(groups_idx[i].end(), centroid_idx_copy.begin(), centroid_idx_copy.end());
        }
        assert(groups_idx[i].size() == _max_pts);
    }

    return groups_idx;
}

TORCH_MODULE(SetAbstraction);

class PointNet2BackboneImpl : public torch::nn::Module {
public:
    PointNet2BackboneImpl(int64_t _num_input_feature, const torch::Tensor _groups_param);
    ~PointNet2BackboneImpl();
    torch::Tensor forward(torch::Tensor _x);

    std::vector<torch::Tensor> skip_features;

private:
    torch::nn::ModuleList set_abstractions_;
    torch::Tensor groups_param_;
};

PointNet2BackboneImpl::PointNet2BackboneImpl(int64_t _num_input_feature, const torch::Tensor _groups_param)
    : groups_param_(_groups_param)
{
    if (groups_param_.size(-1) != 4)
        std::runtime_error("coulumn of groups_param_ tensor must be 4."); // groups_param_ = [int64_t _num_group, float _nn_radius, int64_t _num_pts, int64_t _num_global_feature]

    for (int i = 0; i < groups_param_.size(-2); i++) {
        // if (i = 0) {
        //     set_abstractions_->push_back(SetAbstraction(_num_input_feature, groups_param_.index({ "...", i, 0 }).item<int64_t>(), groups_param_.index({ "...", i, 1 }).to(torch::kFloat).item<float>(), groups_param_.index({ "...", i, 2 }).item<int64_t>(), groups_param_.index({ "...", i, 3 }).item<int64_t>()));
        // } else {
        //     set_abstractions_->push_back(SetAbstraction(groups_param_.index({ "...", groups_param_.size(-2) - i - 1, -1 }).item<int64_t>(), groups_param_.index({ "...", i, 0 }).item<int64_t>(), groups_param_.index({ "...", i, 1 }).to(torch::kFloat).item<float>(), groups_param_.index({ "...", i, 2 }).item<int64_t>(), groups_param_.index({ "...", i, 3 }).item<int64_t>()));
        // }
        set_abstractions_->push_back(SetAbstraction(3, groups_param_.index({ "...", i, 0 }).item<int64_t>(), groups_param_.index({ "...", i, 1 }).to(torch::kFloat).item<float>(), groups_param_.index({ "...", i, 2 }).item<int64_t>(), groups_param_.index({ "...", i, 3 }).item<int64_t>()));
    }
    register_module("set_abstractions_", set_abstractions_);
}

PointNet2BackboneImpl::~PointNet2BackboneImpl()
{
}

torch::Tensor PointNet2BackboneImpl::forward(torch::Tensor _x)
{
    std::cout << "PointNet2Backbone::_x[size]: " << _x.sizes() << std::endl;

    for (auto seq : *set_abstractions_) {
        skip_features.push_back(_x.clone());
        _x = seq->as<SetAbstraction>()->forward(_x);
        std::cout << "_x[size]: " << _x.sizes() << std::endl;
    }

    return _x;
}

TORCH_MODULE(PointNet2Backbone);

class PointNet2SegmentationImpl
    : public torch::nn::Module {
public:
    PointNet2SegmentationImpl(int64_t _num_input_feature, const torch::Tensor _groups_param, int64_t _num_class);
    ~PointNet2SegmentationImpl();
    torch::Tensor forward(torch::Tensor _x);
    torch::Tensor featurePropogate(int64_t _level, torch::Tensor _feature_pt, torch::Tensor _skip_connection, const int64_t _num_nn = 3, const int64_t _lp_norm_order = 2) const;

private:
    torch::nn::ModuleList backbone_, seg_head_;
    int64_t num_class_;
    std::vector<int64_t> decoder_feature_dims_;
};

PointNet2SegmentationImpl::PointNet2SegmentationImpl(int64_t _num_input_feature, const torch::Tensor _groups_param, int64_t _num_class)
    : num_class_(_num_class)
{
    /*Backbone*/
    backbone_->push_back(PointNet2Backbone(_num_input_feature, _groups_param));
    register_module("backbone_", backbone_);

    /*Segmentation Head*/
    // seg_head_->push_back(PointNetBackbone(_groups_param.index({ _groups_param.size(-2) - 2, -1 }).item<int64_t>(), 1, _num_class));
    seg_head_->push_back(PointNetBackbone(3, 1, _num_class));
    decoder_feature_dims_.push_back(_num_class);
    for (int i = 0; i < _groups_param.size(-2) - 1; i++) {
        std::cout << "input_feature[size]: " << _groups_param.index({ _groups_param.size(-2) - i - 1, -1 }).item<int64_t>() << std::endl;
        // seg_head_->push_back(PointNetBackbone(_groups_param.index({ _groups_param.size(-2) - i - 1, -1 }).item<int64_t>(), 1, _groups_param.index({ "...", i, -1 }).item<int64_t>()));
        seg_head_->push_back(PointNetBackbone(3, 1, _groups_param.index({ "...", i, -1 }).item<int64_t>()));
        decoder_feature_dims_.push_back(_groups_param.index({ "...", i, -1 }).item<int64_t>());
    }
    std::cout << "decoder_feature_dims_: " << torch::from_blob(decoder_feature_dims_.data(), { decoder_feature_dims_.size() }, torch::TensorOptions().dtype(torch::kLong)).unsqueeze(0) << std::endl;
    register_module("seg_head_", seg_head_);
}

PointNet2SegmentationImpl::~PointNet2SegmentationImpl()
{
}

torch::Tensor PointNet2SegmentationImpl::forward(torch::Tensor _x)
{
    /*BackBone*/
    _x = backbone_->operator[](0)->as<PointNet2Backbone>()->forward(_x);
    for (const auto& feat : backbone_->operator[](0)->as<PointNet2Backbone>()->skip_features) {
        std::cout << "skip_features_[size]: " << feat.sizes() << std::endl;
    }

    /*Feature Propogation*/
    std::cout << "_x[size]: " << _x.sizes() << std::endl;
    std::cout << "num_skip: " << backbone_->operator[](0)->as<PointNet2Backbone>()->skip_features.size() << std::endl;
    auto skip_feature_it = backbone_->operator[](0)->as<PointNet2Backbone>()->skip_features.rbegin();
    while (skip_feature_it != backbone_->operator[](0)->as<PointNet2Backbone>()->skip_features.rend()) {
        std::cout << "skip[size]: " << skip_feature_it->sizes() << std::endl;
        const auto level = std::distance(skip_feature_it, backbone_->operator[](0)->as<PointNet2Backbone>()->skip_features.rend() - 1);
        std::cout << "level: " << level << std::endl;
        _x = featurePropogate(level, _x, *skip_feature_it);
        std::cout << "_x[size]: " << _x.sizes() << std::endl;
        ++skip_feature_it;
    }

    return _x.index({ "...", Slice(), Slice(3) });
}

torch::Tensor PointNet2SegmentationImpl::featurePropogate(int64_t _level, torch::Tensor _x, torch::Tensor _skip_connection, const int64_t _num_nn, const int64_t _lp_norm_order) const
{
    if (_x.size(-1) < 3 || _skip_connection.size(-1) < 3)
        std::runtime_error("feature dimension must be at least 3");
    if (_x.size(-2) < _num_nn || _skip_connection.size(-2) < _num_nn)
        std::runtime_error("number of nearest neighbor has exceed number of samples.");

    int64_t num_skip_feat = _skip_connection.size(-1) - 3;
    auto out = torch::empty({ _skip_connection.size(0), _skip_connection.size(-2), 3 + decoder_feature_dims_[_level] }, torch::TensorOptions().device(_x.device()));
    auto interpolated_features = torch::empty({ _skip_connection.size(0), _skip_connection.size(-2), _x.size(-1) + _skip_connection.size(-1) - 6 }, torch::TensorOptions().device(_x.device()));

    /*Feature Interpolation*/
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    for (int batch = 0; batch < _skip_connection.size(0); batch++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_xyz;
        input_xyz = tensorToPointCloud<pcl::PointXYZ>(_x.index({ batch, Slice(None, 3, None) }).to(torch::kCPU));
        for (int group = 0; group < _skip_connection.size(-2); group++) {
            /*Nearest Neighbor Search*/
            std::vector<int> indices(_num_nn);
            std::vector<float> distances(_num_nn);
            auto query_tensor = _skip_connection.index({ batch, group, Slice(None, 3, None) }).to(torch::kCPU);
            auto query_pt = pcl::PointXYZ();
            query_pt.getVector3fMap() = Eigen::Map<Eigen::Vector3f>(query_tensor.data_ptr<float>(), 3);
            kdtree.setInputCloud(input_xyz);
            int nn_num = kdtree.nearestKSearch(query_pt, _num_nn, indices, distances);
            if (nn_num > 0) {
                out.index_put_({ batch, group, Slice(None, 3, None) }, query_tensor);
                if (num_skip_feat > 0) {
                    torch::Tensor idx_tensor = torch::from_blob(indices.data(), { indices.size() }, torch::TensorOptions().dtype(torch::kInt)).to(torch::kLong).to(_x.device());
                    auto nn_features = _x.index_select(-2, idx_tensor).index({ batch, "...", Slice(3) }).clone().to(torch::kCPU);
                    torch::Tensor inv_dist_weights = torch::from_blob(distances.data(), { distances.size() });
                    if (inv_dist_weights.select(-1, 0).item<float>() == 0.f) {
                        inv_dist_weights.add_(1e-5);
                    }
                    inv_dist_weights.reciprocal_();
                    interpolated_features.index_put_({ batch, group, Slice(None, nn_features.size(-1)) }, inv_dist_weights.matmul(nn_features).div(inv_dist_weights.sum()));
                    interpolated_features.index_put_({ batch, group, Slice(nn_features.size(-1)) }, _skip_connection.index({ batch, group, Slice(3) }));
                }
            }
        }
    }

    /*Unit PointNet*/
    for (int group = 0; group < _skip_connection.size(-2); group++) {
        seg_head_->operator[](_level)->as<PointNetBackbone>()->to(_x.device());
        auto pointnet_out = std::get<0>(seg_head_->operator[](_level)->as<PointNetBackbone>()->forward(out.index({ "...", group, Slice(None, 3) }).unsqueeze(1).transpose(-2, -1)));
        out.index_put_({ "...", group, Slice(3) }, pointnet_out);
    }

    return out;
}

TORCH_MODULE(PointNet2Segmentation);

} // namespace pointnet

#endif