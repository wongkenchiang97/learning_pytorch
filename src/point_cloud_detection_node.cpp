#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pytorch_tutorial/point_cloud_detection/pointnet.h>
#include <ros/ros.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_detection");
    ros::NodeHandle nh;

    auto device = (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "device: " << (torch::cuda::is_available() ? "true." : "false.") << std::endl;

    /*Test data*/
    const int batch_size = 2;
    const int num_pts = 1000;
    auto points = torch::rand({ batch_size, 3, num_pts }, torch::requires_grad(true).device(device));
    std::cout << "points[device]: " << points.device() << std::endl;
    std::cout << "points[size]: " << points.sizes() << std::endl;
    auto feature = torch::rand({ batch_size, 64, num_pts }, torch::requires_grad(true).device(device));
    std::cout << "feature[device]: " << feature.device() << std::endl;
    std::cout << "feature[size]: " << feature.sizes() << std::endl;


    /*Model*/
    const int num_global_feature = 1024;
    const int num_class = 5;
    auto tnet3 = TNet(3, num_pts);
    tnet3->to(device);
    auto tnet64 = TNet(64, num_pts);
    tnet64->to(device);
    auto backbone = PointNetBackbone(num_pts, num_global_feature,false);
    backbone->to(device);
    auto classification_net = PointNetSegmentation(num_pts, num_global_feature, num_class);
    classification_net->to(device);
    auto segmentation_net = PointNetSegmentation(num_pts, num_global_feature, num_class);
    segmentation_net->to(device);

    auto tnet3_out = tnet3->forward(points);
    std::cout << "tnet3_out[size]: " << tnet3_out.sizes() << std::endl;
    auto tnet64_out = tnet64->forward(feature);
    std::cout << "tnet64_out[size]: " << tnet64_out.sizes() << std::endl;
    auto backbone_out = backbone->forward(points);
    std::cout << "backbone_out[0][size]: " << std::get<0>(backbone_out).sizes() << std::endl;
    std::cout << "backbone_out[0][device]: " << std::get<0>(backbone_out).device() << std::endl;
    std::cout << "backbone_out[1][size]: " << std::get<1>(backbone_out).sizes() << std::endl;
    std::cout << "backbone_out[1][device]: " << std::get<1>(backbone_out).device() << std::endl;
    std::cout << "backbone_out[2][size]: " << std::get<2>(backbone_out).sizes() << std::endl;
    std::cout << "backbone_out[2][device]: " << std::get<2>(backbone_out).device() << std::endl;
    auto class_out = classification_net->forward(points);
    std::cout << "class_out[0][size]: " << std::get<0>(class_out).sizes() << std::endl;
    std::cout << "class_out[1][size]: " << std::get<1>(class_out).sizes() << std::endl;
    std::cout << "class_out[2][size]: " << std::get<2>(class_out).sizes() << std::endl;
    auto seg_out = segmentation_net->forward(points);
    std::cout << "seg_out[size]: " << std::get<0>(seg_out).sizes() << std::endl;
    std::cout << "seg_out[0][size]: " << std::get<0>(seg_out).sizes() << std::endl;
    std::cout << "seg_out[1][size]: " << std::get<1>(seg_out).sizes() << std::endl;
    std::cout << "seg_out[2][size]: " << std::get<2>(seg_out).sizes() << std::endl;

    ros::waitForShutdown();

    return true;
}