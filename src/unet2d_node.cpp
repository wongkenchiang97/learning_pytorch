#include <iostream>
#include <torch/torch.h>
#include <pytorch_tutorial/unet_2D_semantic_segmentation/Unet2D.h>

int main(int argc, char **argv)
{

    auto cuda_available = torch::cuda::is_available();
    auto device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

    /*Hyperparams*/
    const int64_t input_channel = 3;
    const int64_t output_channel = 1;
    const int64_t batch_size = 1;
    const size_t num_epochs = 5;
    const double learning_rate = 1e-3;

    /*UNet 2D NeuralNet*/
    std::vector<int64_t> features = {64, 128, 256, 512};
    torch::Tensor imgs = torch::randn({batch_size, input_channel, 200, 200});
    auto unet2d = Unet2D(input_channel, output_channel, features);
    auto preds = unet2d->forward(imgs);
    std::cout << "size[imgs]: " << imgs.sizes() << std::endl;
    std::cout << "size[preds]: " << preds.sizes() << std::endl;

    return 0;
}