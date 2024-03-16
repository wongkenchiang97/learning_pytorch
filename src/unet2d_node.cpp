#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/indeterminate_progress_bar.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/termcolor.hpp>
#include <iostream>
#include <pytorch_tutorial/unet_2D_semantic_segmentation/Unet2D.h>
#include <pytorch_tutorial/unet_2D_semantic_segmentation/carvana_image_mask_dataset.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <torch/torch.h>

int main(int argc, char** argv)
{

    ros::init(argc, argv, "unet_2d");
    ros::NodeHandle nh;
    auto cuda_available = torch::cuda::is_available();
    auto device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

    /*Hyperparams*/
    const int64_t input_channel = 3;
    const int64_t output_channel = 1;
    const int64_t batch_size = 1;
    const size_t num_epochs = 5;
    const double learning_rate = 1e-6;
    const bool train = true;
    auto read_model_path = ros::package::getPath("pytorch_tutorial") + "/data/model/cavana/model_3000_obj.pth";
    auto save_model_path = ros::package::getPath("pytorch_tutorial") + "/data/model/cavana/test_model.pth";

    std::vector<int64_t> features = { 64, 128, 256, 512 };
    auto unet2d = Unet2D(input_channel, output_channel, features);
    ROS_INFO("loading pre-train model from %s", read_model_path.c_str());
    torch::load(unet2d, read_model_path);
    unet2d->to(device);

    std::string dataset_path = "/home/dylan/Documents/datasets/carvana-image-masking-challenge";

    /*Training Dataset*/
    float mean(0.5f), stdev(2.f);
    ROS_INFO("loading train_dataset...");
    auto train_dataset = carvana_dataset::ImageMask(dataset_path, carvana_dataset::ImageMask::Mode::kTrain)
                             .map(torch::data::transforms::Normalize<>({ mean, mean, mean }, { stdev, stdev, stdev }))
                             .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);
    // auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);
    ROS_INFO("num_train_samples: %ld.", train_dataset.size().value());

    /*Testing Dataset*/
    ROS_INFO("loading train_dataset...");
    auto test_dataset = carvana_dataset::ImageMask(dataset_path, carvana_dataset::ImageMask::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 2.0, 2.0, 2.0 }))
                            .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);
    // auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_dataset), batch_size);
    ROS_INFO("num_test_samples: %ld.", test_dataset.size().value());

    /*Optimizer*/
    torch::optim::Adam optimizer(unet2d->parameters(), torch::optim::AdamOptions(learning_rate));

    /*Loss*/
    auto criterion = torch::nn::BCEWithLogitsLoss();

    /*Train*/
    if (train) {
        using namespace indicators;
        ROS_INFO("Training neural network...");
        BlockProgressBar bar {
            option::BarWidth { 80 },
            option::Start { "[" },
            option::End { "]" },
            option::ForegroundColor { Color::white },
            option::FontStyles { std::vector<FontStyle> { FontStyle::bold } },
        };
        auto progress = 0.0f;
        bar.set_option(option::PostfixText { "[iter 1/" + std::to_string(train_dataset.size().value()) + "]" + "[epoch: 1/" + std::to_string(num_epochs + 1) + "]" });
        indicators::show_console_cursor(false);
        auto num_train_samples = train_dataset.size().value();
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            // Initialize running metrics
            double running_loss = 0.0;
            int64_t iter = 0;
            size_t num_correct = 0;
            size_t accum_num_data = 0;
            for (auto& batch : *train_loader) {
                /*Forward*/
                bar.set_progress(progress);
                auto data = batch.data.to(device);
                auto target = batch.target.to(device).unsqueeze(1);
                auto output = unet2d->forward(data);

                // Update running loss
                auto loss = criterion(output, target.resize_(output.sizes()));

                // Backward and optimize
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                /*Update progress*/
                progress += (float)data.size(0) / (float)num_train_samples * 100.f;
                iter++;
                running_loss += loss.item<double>() * data.size(0);
                accum_num_data += data.size(0);
                auto sample_mean_loss = running_loss / accum_num_data;
                bar.set_option(option::PostfixText { "[iter " + std::to_string(iter) + "/" + std::to_string((int64_t)std::ceil((float)train_dataset.size().value() / (float)batch_size)) + "]" + "[epoch " + std::to_string(epoch + 1) + "/" + std::to_string(num_epochs) + "]" + "[loss: " + std::to_string(sample_mean_loss) + "]" });
            }
            progress = 0.0f;
            auto accuracy = static_cast<double>(num_correct) / num_train_samples;

            /*Save model state*/
            torch::serialize::OutputArchive out;
            unet2d->save(out);
            if (!save_model_path.empty()) {
                out.save_to(save_model_path);
            } else {
                throw std::runtime_error("model path can't be empty");
            }

            // std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            //           << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
        }
        bar.mark_as_completed();
        std::cout << termcolor::bold << termcolor::green
                  << "✔ training process completed."
                  << termcolor::reset;
        indicators::show_console_cursor(true);
    }

    /*load model*/
    auto infer_model = Unet2D(input_channel, output_channel, features);
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
    cv::Mat combine_img;
    cv::Mat combine_horizon_img;
    unsigned int batch_idx = 0;
    for (auto& batch : *test_loader) {
        /*Forward*/
        auto data = batch.data.to(device);
        auto target = batch.target.to(device).unsqueeze(1);
        auto prediction = infer_model->forward(data);
        for (int im = 0; im < prediction.size(0); im++) {
            auto loss = criterion(prediction.index({ im, "..." }), target.index({
                                                                                    im,
                                                                                    "...",
                                                                                })
                                                                       .clone()
                                                                       .resize_(prediction.index({ im, "..." }).sizes()));
            std::cout << "loss[" << batch_size * batch_idx + im << "]: " << loss.item<double>() << std::endl;
            auto data_im = carvana_dataset::ImageMask::TensorToCVColor(data.index({
                                                                                      im,
                                                                                      "...",
                                                                                  })
                                                                           .clone()
                                                                           .squeeze(0)
                                                                           .to(torch::kCPU),
                mean, stdev);
            auto target_mask = carvana_dataset::ImageMask::TensorToCVMask(target.index({
                                                                                           im,
                                                                                           "...",
                                                                                       })
                                                                              .clone()
                                                                              .squeeze(0)
                                                                              .squeeze(0)
                                                                              .to(torch::kCPU),
                mean, stdev);
            cv::cvtColor(target_mask, target_mask, cv::COLOR_BGR2RGB);
            auto pred_im = carvana_dataset::ImageMask::TensorToCVMask(prediction.index({ im, "..." }).clone().to(torch::kCPU).squeeze(0).squeeze(0), mean, stdev);
            cv::resize(pred_im, pred_im, data_im.size());
            cv::cvtColor(pred_im, pred_im, cv::COLOR_BGR2RGB);
            cv::hconcat(data_im, target_mask, combine_img);
            cv::hconcat(combine_img, pred_im, combine_img);
            cv::imshow("prediction.", combine_img);
            cv::waitKey(0);
        }
        batch_idx++;
    }

    // cv::imshow("train_data_and_target.",combine_img);

    ros::waitForShutdown();

    return 0;
}