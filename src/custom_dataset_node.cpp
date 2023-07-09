#include <pytorch_tutorial/cats_and_dogs/dataset.h>

cv::Mat tensorToCV(torch::Tensor _x);

int main(int argc, char **argv)
{
    std::string dataset_path = "/home/dylan/Documents/datasets/cats_and_dogs";
    int64_t batch_size = 4;

    /*create train dataset*/
    auto train_dataset = CatsDogs(dataset_path)
                             .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                             .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);

    /*create test dataset*/
    auto test_dataset = CatsDogs(dataset_path, CatsDogs::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                            .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);

    for (auto &batch : *train_loader)
    {
        auto img = batch.data;
        auto target = batch.target;
        auto out = tensorToCV(img[0]);
        cv::imshow("Display", out);
        int k = cv::waitKey(0);
        break;
    }

    return 0;
}

cv::Mat tensorToCV(torch::Tensor _x)
{
    _x = _x.permute({1, 2, 0});
    _x = _x.mul(0.5).add(0.5).mul(255).clamp(0, 255).to(torch::kByte);
    _x = _x.contiguous();

    int height = _x.size(0);
    int width = _x.size(1);
    cv::Mat output(cv::Size(width, height), CV_8UC3);
    std::memcpy((void *)output.data, _x.data_ptr(), sizeof(torch::kU8) * _x.numel());
    cv::cvtColor(output, output, cv::COLOR_BGR2RGB);

    return output.clone();
}
