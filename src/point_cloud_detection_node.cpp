#define PCL_NO_PRECOMPILE
#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pytorch_tutorial/point_cloud_detection/point_type.h>
#include <pytorch_tutorial/point_cloud_detection/pointnet.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

void getPointCloudFromLabel(pcl::PointCloud<pcl_ext::PointXYZLO>::ConstPtr _in, const std::vector<int> _labels, pcl::PointCloud<pcl_ext::PointXYZLO>& _out)
{
    /*Set condition*/
    pcl::ConditionOr<pcl_ext::PointXYZLO>::Ptr label_cond;
    label_cond = pcl::make_shared<pcl::ConditionOr<pcl_ext::PointXYZLO>>();
    for (const auto label : _labels) {
        label_cond->addComparison(pcl::FieldComparison<pcl_ext::PointXYZLO>::Ptr(new pcl::FieldComparison<pcl_ext::PointXYZLO>("label", pcl::ComparisonOps::EQ, label)));
    }

    /*Set condition filter*/
    pcl::ConditionalRemoval<pcl_ext::PointXYZLO> label_filt(true);
    label_filt.setInputCloud(_in);
    label_filt.setCondition(label_cond);
    label_filt.setKeepOrganized(false);
    label_filt.filter(_out);
}

void getRandomSample(pcl::PointCloud<pcl_ext::PointXYZLO>::ConstPtr _in, const int64_t _num_sample, pcl::PointCloud<pcl_ext::PointXYZLO>& _out)
{
    pcl::RandomSample<pcl_ext::PointXYZLO> sample(true);
    sample.setInputCloud(_in);
    sample.setSample(_num_sample);
    pcl::Indices object_indices;
    pcl::PointCloud<pcl_ext::PointXYZLO> sample_tmp;
    sample.filter(sample_tmp);
    sample_tmp.swap(_out);
}

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
    auto read_model_path = ros::package::getPath("pytorch_tutorial") + "/data/model/pointnet/pallet_noise_segmentation/overfit_test_model.pth";
    auto save_model_path = ros::package::getPath("pytorch_tutorial") + "/data/model/pointnet/pallet_noise_segmentation/test_model.pth";

    std::cout << "device: " << (torch::cuda::is_available() ? "true." : "false.") << std::endl;

    /*Test data*/
    auto points = torch::rand({ batch_size, 3, max_num_pts }, torch::requires_grad(true).device(device));
    std::cout << "points[size]: " << points.sizes() << std::endl;
    std::string pcd = "/home/dylan/Documents/datasets/pallet3d/usun_europallet/train/target/0.pcd";
    pcl::PointCloud<pcl_ext::PointXYZLO>::Ptr data_cloud, target_cloud, sampled_cloud;
    data_cloud = pcl::make_shared<pcl::PointCloud<pcl_ext::PointXYZLO>>();
    target_cloud = pcl::make_shared<pcl::PointCloud<pcl_ext::PointXYZLO>>();
    sampled_cloud = pcl::make_shared<pcl::PointCloud<pcl_ext::PointXYZLO>>();
    pcl::io::loadPCDFile(pcd, *data_cloud);

    std::vector<int> labels = { 1, 2 };
    getPointCloudFromLabel(data_cloud, labels, *target_cloud);
    sensor_msgs::PointCloud2 data_out, target_out, sampled_out;
    pcl::toROSMsg(*data_cloud, data_out);
    data_out.header.frame_id = "camera";
    data_out.header.stamp = ros::Time::now();
    data_cloud_pub.publish(data_out);
    pcl::toROSMsg(*target_cloud, target_out);
    target_out.header.frame_id = "camera";
    target_out.header.stamp = ros::Time::now();
    target_cloud_pub.publish(target_out);
    pcl::PointCloud<pcl_ext::PointXYZLO> sample_tmp;
    getRandomSample(target_cloud, 3000, sample_tmp);
    sampled_cloud->operator+=(sample_tmp);
    getRandomSample(data_cloud, 2000, sample_tmp);
    sampled_cloud->operator+=(sample_tmp);
    pcl::toROSMsg(*sampled_cloud, sampled_out);
    sampled_out.header.frame_id = "camera";
    sampled_out.header.stamp = ros::Time::now();
    sampled_cloud_pub.publish(sampled_out);
    auto label_obj_map = Eigen::Map<Eigen::MatrixXi, 0, Eigen::Stride<sizeof(pcl_ext::PointXYZLO) / sizeof(int), 1>>(&sampled_cloud->points[0].label, 2, sampled_cloud->points.size());
    auto pts_map = Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<sizeof(pcl_ext::PointXYZLO) / sizeof(float), 1>>(&sampled_cloud->points[0].x, 3, sampled_cloud->points.size());
    Eigen::MatrixXf xyz = pts_map;
    Eigen::VectorXi label = label_obj_map.row(0);
    // std::cout << "label:\n"
    //           << label.transpose() << std::endl;
    torch::Tensor tensor_target = torch::from_blob(label.data(), label.rows());
    tensor_target = tensor_target.to(torch::kLong).unsqueeze(0).to(device);
    // std::cout << "tensor_target:\n"
    //           << tensor_target << std::endl;
    torch::Tensor tensor_data = torch::from_blob(xyz.data(), { 3, xyz.cols() }, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat));
    tensor_data = tensor_data.view({ xyz.cols(), 3 }).transpose(1, 0).to(device).unsqueeze(0);
    if (batch_size > 1) {
        tensor_target = tensor_target.repeat({ batch_size, 1 });
        tensor_data = tensor_data.repeat({ batch_size, 1, 1 });
    }

    std::cout << "tensor_target[size]: " << tensor_target.sizes() << std::endl;
    const int num_class = label_obj_map.row(0).maxCoeff() + 1;
    std::cout << "num_class: " << num_class << std::endl;
    auto label_one_hot = torch::zeros({ label_obj_map.cols(), num_class }, torch::TensorOptions().device(device));
    for (int i = 0; i < label_obj_map.cols(); i++) {
        label_one_hot.index_put_({ i, label_obj_map(0, i) }, 1.f);
    }
    label_one_hot = label_one_hot.repeat({ batch_size, 1, 1 });
    std::cout << "tensor_data[size]: " << tensor_data.sizes() << std::endl;
    std::cout << "label_one_hot[size]: " << label_one_hot.sizes() << std::endl;

    /*Model*/
    const int num_global_feature = 1024;
    auto segmentation_net = PointNetSegmentation(max_num_pts, num_global_feature, num_class);
    segmentation_net->to(device);

    /*Optimizer*/
    torch::optim::Adam optimizer(segmentation_net->parameters(), torch::optim::AdamOptions(learning_rate));

    /*Loss*/
    auto criterion = torch::nn::CrossEntropyLoss();

    /*Train*/
    if (train) {
        auto identity = torch::eye(64, torch::TensorOptions().device(device)).repeat({ batch_size, 1, 1 });
        ROS_INFO("training...");
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            /*Forward*/
            auto seg_out = segmentation_net->forward(tensor_data);

            /*loss Criteria*/
            auto loss_feature = criterion(std::get<0>(seg_out).transpose(2, 1), tensor_target);
            // auto loss_feature = criterion(std::get<0>(seg_out).transpose(2, 1).softmax(1), label_one_hot);

            ROS_INFO("loss: %f", loss_feature.item<double>());

            /*Backward and optimize*/
            optimizer.zero_grad();
            loss_feature.backward();
            optimizer.step();
        }
        ROS_INFO("finish training.");

        /*Save model state*/
        torch::serialize::OutputArchive out;
        segmentation_net->save(out);
        if (!save_model_path.empty()) {
            out.save_to(save_model_path);
        } else {
            throw std::runtime_error("model path can't be empty");
        }
    }

    /*load model*/
    auto infer_model = PointNetSegmentation(max_num_pts, num_global_feature, num_class);
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
    getRandomSample(data_cloud, max_num_pts, inference_cloud);
    std::cout << "inference_cloud[size]: " << inference_cloud.size() << std::endl;
    auto obj_map = Eigen::Map<Eigen::MatrixXi, 0, Eigen::Stride<sizeof(pcl_ext::PointXYZLO) / sizeof(int), 1>>(&inference_cloud.points[0].label, 2, inference_cloud.points.size());
    std::cout << "label[sum]: " << obj_map.row(0).sum() << std::endl;
    obj_map.row(0) = Eigen::VectorXi::Zero(obj_map.cols()).transpose();
    std::cout << "label[sum]: " << obj_map.row(0).sum() << std::endl;
    auto inf_pts_map = Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<sizeof(pcl_ext::PointXYZLO) / sizeof(float), 1>>(&sampled_cloud->points[0].x, 3, sampled_cloud->points.size());
    Eigen::MatrixXf inf_xyz = inf_pts_map;
    torch::Tensor inf_data = torch::from_blob(xyz.data(), { 3, xyz.cols() }, torch::TensorOptions().requires_grad(true).dtype(torch::kFloat));
    inf_data = inf_data.view({ xyz.cols(), 3 }).transpose(1, 0).to(device).unsqueeze(0);

    auto prediction = infer_model->forward(tensor_data);

    ros::waitForShutdown();

    return true;
}