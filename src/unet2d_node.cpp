#include <iostream>
#include <torch/torch.h>
#include <pytorch_tutorial/unet_2D_semantic_segmentation/Unet2D.h>
#include <pytorch_tutorial/unet_2D_semantic_segmentation/carvana_image_mask_dataset.h>
template <typename Loader>
void showLoaderData(Loader _loader);

int main(int argc, char **argv)
{

    auto cuda_available = torch::cuda::is_available();
    auto device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

    /*Hyperparams*/
    const int64_t input_channel = 3;
    const int64_t output_channel = 1;
    const int64_t batch_size = 4;
    const size_t num_epochs = 5;
    const double learning_rate = 1e-3;

    /*UNet 2D NeuralNet*/
    std::vector<int64_t> features = {64, 128, 256, 512};
    torch::Tensor imgs = torch::randn({batch_size, input_channel, 200, 200});
    auto unet2d = Unet2D(input_channel, output_channel, features);
    // auto preds = unet2d->forward(imgs);
    // std::cout << "size[imgs]: " << imgs.sizes() << std::endl;
    // std::cout << "size[preds]: " << preds.sizes() << std::endl;

    std::string dataset_path = "/home/dylan/Documents/datasets/carvana-image-masking-challenge";

    /*Training Dataset*/
    auto train_dataset = carvana_dataset::ImageMask(dataset_path, carvana_dataset::ImageMask::Mode::kTest)
                             .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                             .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);

    /*Testing Dataset*/
    auto test_dataset = carvana_dataset::ImageMask(dataset_path, carvana_dataset::ImageMask::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                            .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);

    /*show loader data*/
    showLoaderData(train_dataset);

    /*Optimizer*/
    torch::optim::Adam optimizer(unet2d->parameters(), torch::optim::AdamOptions(learning_rate));

    return 0;
}

template <typename Loader>
void showLoaderData(Loader _loader)
{
    // int64_t batch_idx = 0;
    // for (auto batch : _loader)
    // {
    //     std::cout << "batch[" << batch_idx << "]" << std::endl;
    //     for (int i = 0; i < batch.data.sizes().at(0); i++)
    //     {
    //         auto tensor_img = batch.data[i];
    //         auto tensor_mask = batch.target[i];

    //         auto img = carvana_dataset::ImageMask::TensorToCVColor(tensor_img);
    //         auto mask = carvana_dataset::ImageMask::TensorToCVMask(tensor_mask);

    //         cv::Mat dst;
    //         double alpha = 0.5;
    //         double beta = 1 - alpha;
    //         cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
    //         cv::applyColorMap(mask, mask, cv::COLORMAP_TWILIGHT_SHIFTED);
    //         cv::addWeighted(img, alpha, mask, beta, 0.0, dst);
    //         cv::hconcat(dst, mask, dst);
    //         imshow("loader data", dst);
    //         cv::waitKey(3e3);
    //     }
    //     batch_idx++;
    // }
}