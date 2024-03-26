#define PCL_NO_PRECOMPILE
#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pytorch_tutorial/point_cloud_detection/point_type.h>
#include <pytorch_tutorial/point_cloud_detection/pointnet.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <torch/linalg.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

template <typename PointT>
void getPointCloudFromLabel(typename pcl::PointCloud<PointT>::ConstPtr _in, const std::vector<int> _labels, pcl::PointCloud<PointT>& _out)
{
    /*Set condition*/
    typename pcl::ConditionOr<PointT>::Ptr label_cond;
    label_cond = pcl::make_shared<pcl::ConditionOr<PointT>>();
    for (const auto label : _labels) {
        label_cond->addComparison(typename pcl::FieldComparison<PointT>::Ptr(new pcl::FieldComparison<PointT>("label", pcl::ComparisonOps::EQ, label)));
    }

    /*Set condition filter*/
    typename pcl::ConditionalRemoval<PointT> label_filt(true);
    label_filt.setInputCloud(_in);
    label_filt.setCondition(label_cond);
    label_filt.setKeepOrganized(false);
    label_filt.filter(_out);
}

template <typename PointT>
void getPointCloudFromDepth(typename pcl::PointCloud<PointT>::ConstPtr _in, float _depth, pcl::PointCloud<PointT>& _out)
{
    /*Set condition*/
    typename pcl::ConditionOr<PointT>::Ptr label_cond;
    label_cond = pcl::make_shared<pcl::ConditionOr<PointT>>();
    label_cond->addComparison(typename pcl::FieldComparison<PointT>::Ptr(new pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::LT, _depth)));

    /*Set condition filter*/
    typename pcl::ConditionalRemoval<PointT> label_filt(true);
    label_filt.setInputCloud(_in);
    label_filt.setCondition(label_cond);
    label_filt.setKeepOrganized(false);
    label_filt.filter(_out);
}

template <typename PointT>
void getRandomSample(typename pcl::PointCloud<PointT>::ConstPtr _in, const int64_t _num_sample, pcl::PointCloud<PointT>& _out)
{
    typename pcl::RandomSample<PointT> sample(true);
    sample.setInputCloud(_in);
    sample.setSample(_num_sample);
    pcl::Indices object_indices;
    typename pcl::PointCloud<PointT> sample_tmp;
    sample.filter(sample_tmp);
    sample_tmp.swap(_out);
}

template <typename PointT>
void publishPointCloud(const std::string _topic, const pcl::PointCloud<PointT>& _cloud, const ros::Publisher& pub, const ros::Time _stamp, const std::string _frame_id)
{
    sensor_msgs::PointCloud2 cloud_out;
    pcl::toROSMsg(_cloud, cloud_out);
    cloud_out.header.frame_id = _frame_id;
    cloud_out.header.stamp = _stamp;
    pub.publish(cloud_out);
}

void setLabelFromPrediction(const torch::Tensor& _prediction, Eigen::VectorXi& _label)
{
    auto prob_map = Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<-1, -1>>(_prediction.data_ptr<float>(), 3, _prediction.size(1), Eigen::Stride<-1, -1>(_prediction.size(0), _prediction.size(1)));
}

enum class ModelTask : int {
    CLASSIFICATION,
    SEGMENTATION
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_detection");
    ros::NodeHandle nh;
    ros::Publisher data_cloud_pub, target_cloud_pub, sampled_cloud_pub, inference_cloud_pub;
    data_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("data_cloud", 1, true);
    target_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("target_cloud", 1, true);
    sampled_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("sampled_cloud", 1, true);
    inference_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("inference_cloud", 1, true);

    auto device = (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    const int batch_size = 2;
    const int max_num_pts = 5000;
    auto learning_rate = 1e-3;
    const int64_t num_epochs = 1000;
    // const int64_t num_epochs = 1;
    const bool train = true;
    ModelTask task = ModelTask::CLASSIFICATION;
    // auto read_model_path = ros::package::getPath("pytorch_tutorial") + "/data/model/pointnet/pallet_noise_segmentation/overfit_test_model.pth";
    auto read_model_path = ros::package::getPath("pytorch_tutorial") + "/data/model/pointnet/pallet_noise_segmentation/overfit_classification_model.pth";
    auto save_model_path = ros::package::getPath("pytorch_tutorial") + "/data/model/pointnet/pallet_noise_segmentation/test_model.pth";
    std::cout << "device: " << (torch::cuda::is_available() ? "true." : "false.") << std::endl;
    switch (task) {
    case ModelTask::CLASSIFICATION: {
        ROS_INFO("task: CLASSIFICATION");
    } break;

    case ModelTask::SEGMENTATION: {
        ROS_INFO("task: SEGMENTATION");
    } break;

    default:
        break;
    }

    /*Test data*/
    std::string pcd = "/home/dylan/Documents/datasets/pallet3d/usun_europallet/train/target/0.pcd";
    // std::string pcd = "/home/dylan/Documents/datasets/obstacle_detection/scene_cloud_18-03-20241858299.pcd";
    pcl::PointCloud<pcl_ext::PointXYZLO>::Ptr data_cloud, target_cloud, sampled_cloud;
    data_cloud = pcl::make_shared<pcl::PointCloud<pcl_ext::PointXYZLO>>();
    target_cloud = pcl::make_shared<pcl::PointCloud<pcl_ext::PointXYZLO>>();
    sampled_cloud = pcl::make_shared<pcl::PointCloud<pcl_ext::PointXYZLO>>();
    pcl::io::loadPCDFile(pcd, *data_cloud);

    // std::vector<int> labels = { 1, 2 };//get pallet & noise cloud
    std::vector<int> labels = { 1 }; // get pallet cloud
    // std::vector<int> labels = { 2, 4 };
    getPointCloudFromLabel<pcl_ext::PointXYZLO>(data_cloud, labels, *target_cloud);
    // getPointCloudFromDepth<pcl_ext::PointXYZLO>(target_cloud, 1.2f, *target_cloud);
    // std::cout << "target_cloud[size]: " << target_cloud->size() << std::endl;
    auto measurement_stamp = ros::Time::now();
    publishPointCloud<pcl_ext::PointXYZLO>("data_cloud", *data_cloud, data_cloud_pub, measurement_stamp, "camera");
    publishPointCloud<pcl_ext::PointXYZLO>("target_cloud", *target_cloud, target_cloud_pub, measurement_stamp, "camera");
    pcl::PointCloud<pcl_ext::PointXYZLO> sample_tmp;
    getRandomSample<pcl_ext::PointXYZLO>(target_cloud, 5000, sample_tmp);
    sampled_cloud->operator+=(sample_tmp);
    // getRandomSample<pcl_ext::PointXYZLO>(data_cloud, 1000, sample_tmp);
    // sampled_cloud->operator+=(sample_tmp);
    publishPointCloud<pcl_ext::PointXYZLO>("sampled_cloud", *sampled_cloud, sampled_cloud_pub, measurement_stamp, "camera");

    auto label_obj_map = Eigen::Map<Eigen::MatrixXi, 0, Eigen::Stride<sizeof(pcl_ext::PointXYZLO) / sizeof(int), 1>>(&sampled_cloud->points[0].label, 2, sampled_cloud->points.size());
    auto pts_map = Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<sizeof(pcl_ext::PointXYZLO) / sizeof(float), 1>>(&sampled_cloud->points[0].x, 3, sampled_cloud->points.size());
    Eigen::MatrixXf xyz = pts_map;
    Eigen::VectorXi label = label_obj_map.row(0);
    // std::cout << "label: " << label.transpose() << std::endl;
    torch::Tensor tensor_target, label_one_hot;
    torch::Tensor tensor_data = torch::from_blob(xyz.data(), { 3, xyz.cols() }, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat)).clone().to(device);
    tensor_data = tensor_data.view({ xyz.cols(), 3 }).transpose(1, 0).to(device).unsqueeze(0);
    auto tensor_label_idx = torch::from_blob(label.data(), label.rows(), torch::TensorOptions().dtype(torch::kInt)).clone().unsqueeze(0).to(torch::kLong).to(device);
    // std::cout << "tensor_label_idx: " << tensor_label_idx << std::endl;
    // std::cout << "tensor_target: " << tensor_target << std::endl;
    const int num_class = label_obj_map.row(0).maxCoeff() + 1;
    std::cout << "num_class: " << num_class << std::endl;
    switch (task) {
    case ModelTask::CLASSIFICATION: {
        auto range = torch::arange(num_class, torch::TensorOptions().dtype(torch::kLong));
        label_one_hot = torch::zeros({ 3 }, torch::TensorOptions().device(device));
        label_one_hot.index_put_({ 1 }, 1);
        label_one_hot = label_one_hot.unsqueeze(0);
        label_one_hot = label_one_hot.repeat({ batch_size, 1 });
        std::cout << "label_one_hot:\n"
                  << label_one_hot << std::endl;
    } break;

    case ModelTask::SEGMENTATION: {
        tensor_target = torch::from_blob(label.data(), label.rows(), torch::TensorOptions().dtype(torch::kInt)).unsqueeze(0).to(torch::kLong).to(device);
        tensor_target = tensor_target.repeat({ batch_size, 1 });
        auto range = torch::arange(label.rows(), torch::TensorOptions().dtype(torch::kLong));
        label_one_hot = torch::zeros({ label.rows(), num_class }, torch::TensorOptions().device(device));
        label_one_hot.view({ -1, num_class }).index_put_({ range, tensor_label_idx.view({ -1 }) }, 1);
        label_one_hot = label_one_hot.unsqueeze(0);
        label_one_hot = label_one_hot.repeat({ batch_size, 1, 1 });
    } break;

    default:
        break;
    }
    if (batch_size > 1) {
        tensor_data = tensor_data.repeat({ batch_size, 1, 1 });
        tensor_label_idx = tensor_label_idx.repeat({ batch_size, 1 });
    }
    std::cout << "tensor_target[size]: " << tensor_target.sizes() << std::endl;
    std::cout << "tensor_data[size]: " << tensor_data.sizes() << std::endl;
    std::cout << "tensor_label_idx[size]: " << tensor_label_idx.sizes() << std::endl;
    std::cout << "label_one_hot[size]: " << label_one_hot.sizes() << std::endl;

    /*Model*/
    const int num_global_feature = 1024;
    // const int num_global_feature = 4096;
    auto classification_net = PointNetClassification(max_num_pts, num_global_feature, 3);
    auto segmentation_net = PointNetSegmentation(max_num_pts, num_global_feature, num_class);
    switch (task) {
    case ModelTask::CLASSIFICATION: {
        classification_net->to(device);
    } break;

    case ModelTask::SEGMENTATION: {
        segmentation_net->to(device);
    } break;

    default:
        break;
    }

    /*Optimizer*/
    // torch::optim::Adam optimizer(segmentation_net->parameters(), torch::optim::AdamOptions(learning_rate));
    torch::optim::Adam optimizer(classification_net->parameters(), torch::optim::AdamOptions(learning_rate));

    /*Loss*/
    auto weights = torch::zeros({ 3 }, torch::TensorOptions().device(device));
    weights.index_put_({ 0 }, 0.01);
    weights.index_put_({ 1 }, 0.3);
    weights.index_put_({ 2 }, 10.7);
    auto ign_idx = torch::zeros({ 1 });
    // std::cout << "weights: " << weights << std::endl;
    // std::cout << "ign_idx: " << ign_idx << std::endl;
    // auto criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().weight(weights));
    auto criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions());

    /*Train*/
    // std::cout << "task: " << task << std::endl;
    if (train) {
        auto identity = torch::eye(64, torch::TensorOptions().device(device)).repeat({ batch_size, 1, 1 });
        ROS_INFO("training...");
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> out;
            torch::Tensor loss_feature;
            switch (task) {
            case ModelTask::CLASSIFICATION: {
                /*Forward*/
                out = classification_net->forward(tensor_data);
                // std::cout << "out:\n"
                //           << std::get<0>(out) << std::endl;

                /*loss Criteria*/
                loss_feature = criterion(std::get<0>(out).transpose(1, 0), label_one_hot.transpose(1, 0));

            } break;

            case ModelTask::SEGMENTATION: {
                /*Forward*/
                out = segmentation_net->forward(tensor_data);

                /*loss Criteria*/
                // auto loss_feature = criterion(std::get<0>(seg_out).transpose(2, 1), tensor_target);
                loss_feature = criterion(std::get<0>(out).transpose(2, 1).softmax(1), label_one_hot.transpose(2, 1));
            } break;

            default:
                break;
            }

            ROS_INFO("loss: %f", loss_feature.item<double>());

            /*Backward and optimize*/
            optimizer.zero_grad();
            loss_feature.backward();
            optimizer.step();

            if (loss_feature.item<double>() < 1e-3)
                break;
        }
        ROS_INFO("finish training.");

        /*Save model state*/
        torch::serialize::OutputArchive out;
        switch (task) {
        case ModelTask::CLASSIFICATION: {
            classification_net->save(out);
        } break;

        case ModelTask::SEGMENTATION: {
            segmentation_net->save(out);
        } break;

        default:
            break;
        }
        if (!save_model_path.empty()) {
            out.save_to(save_model_path);
        } else {
            throw std::runtime_error("model path can't be empty");
        }
    }

    /*load model*/
    auto infer_model = PointNetClassification(max_num_pts, num_global_feature, num_class);
    // auto infer_model = PointNetSegmentation(max_num_pts, num_global_feature, num_class);
    if (train) {
        ROS_INFO("loading model from %s.", save_model_path.c_str());
        torch::load(infer_model, save_model_path);
    } else {
        ROS_INFO("loading model from %s.", read_model_path.c_str());
        torch::load(infer_model, read_model_path);
    }

    infer_model->to(device);
    infer_model->eval();

    /*Inference*/
    torch::NoGradGuard no_grad;
    pcl::PointCloud<pcl_ext::PointXYZLO> inference_cloud;
    pcl::copyPointCloud(*sampled_cloud, inference_cloud);
    auto obj_map = Eigen::Map<Eigen::MatrixXi, 0, Eigen::Stride<sizeof(pcl_ext::PointXYZLO) / sizeof(int), 1>>(&inference_cloud.points[0].label, 2, inference_cloud.points.size());
    auto inf_pts_map = Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<sizeof(pcl_ext::PointXYZLO) / sizeof(float), 1>>(&sampled_cloud->points[0].x, 3, sampled_cloud->points.size());
    Eigen::MatrixXf inf_xyz = inf_pts_map;
    torch::Tensor inf_data = torch::from_blob(inf_xyz.data(), { 3, inf_xyz.cols() }, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat));
    inf_data = inf_data.view({ inf_xyz.cols(), 3 }).transpose(1, 0).to(device).unsqueeze(0);
    // getRandomSample(data_cloud, max_num_pts, inference_cloud);
    std::cout << "inference_cloud[size]: " << inference_cloud.size() << std::endl;
    switch (task) {
    case ModelTask::CLASSIFICATION: {
        ROS_INFO("CLASSIFICATION inference");
        auto result = infer_model->forward(tensor_data);
        std::cout << "result:\n"
                  << std::get<0>(result).softmax(-1) << std::endl;
    } break;

    case ModelTask::SEGMENTATION: {
        ROS_INFO("SEGMENTATION inference");
        std::cout << "label[sum]: " << obj_map.row(0).sum() << std::endl;
        obj_map.row(0) = Eigen::VectorXi::Zero(obj_map.cols()).transpose();
        std::cout << "label[sum]: " << obj_map.row(0).sum() << std::endl;

        auto result = infer_model->forward(inf_data);
        auto prob = std::get<0>(result).transpose(2, 1).softmax(1).index({ 0, "..." }).to(torch::kCPU);
        auto prob_map = Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<-1, -1>>(prob.data_ptr<float>(), 3, prob.size(1), Eigen::Stride<-1, -1>(prob.size(0), prob.size(1)));
        // std::cout << "prob_map:\n"
        //           << prob_map << std::endl;
        // prob_map = obj_map.row(0) = Eigen::Map<Eigen::VectorXi>(prob.data_ptr<int>(), prob.size(0));
        publishPointCloud<pcl_ext::PointXYZLO>("inference_cloud", inference_cloud, inference_cloud_pub, measurement_stamp, "camera");
        std::cout << "prob[size]: " << prob.sizes() << std::endl;
        std::cout << "label[sum]: " << obj_map.row(0).sum() << std::endl;
    } break;

    default:
        break;
    }

    ros::waitForShutdown();

    return true;
}