#include <iostream>
#include <ros/ros.h>
#include <torch/torch.h>
#include <Eigen/Eigen>
#include <ros/package.h>
#include <pytorch_tutorial/model.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feed_forward_node");

    auto cuda_available = torch::cuda::is_available();
    auto device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

    /*Hyperparams*/
    const int64_t input_size = 784;
    const int64_t hidden_size = 500;
    const int64_t num_of_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 5;
    const double learning_rate = 1e-3;

    const std::string MNIST_dataset_path = "/home/dylan/Documents/datasets/MNIST/raw";

    /*MNIST Dataset*/
    auto mean = 0.1307;
    auto std = 0.3081;
    auto train_dataset = torch::data::datasets::MNIST(MNIST_dataset_path)
                             .map(torch::data::transforms::Normalize<>(mean, std))
                             .map(torch::data::transforms::Stack<>());

    /*Num samples in Dataset*/
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(MNIST_dataset_path, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(mean, std))
                            .map(torch::data::transforms::Stack<>());

    auto num_test_samples = test_dataset.size().value();

    /*Data loaders*/
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    /*NeralNet*/
    Model model(input_size, hidden_size, num_of_classes);

    std::cout << "model:\n"
              << model << std::endl;

    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto &batch : *train_loader)
        {
            auto data = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                  << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto &batch : *test_loader)
    {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);

        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

    return 0;
}