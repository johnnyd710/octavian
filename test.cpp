/*
@author: John DiMatteo
creation date: 03-14-2020
desc:
extreme learning machine using Pytorch Tensors

INPUT
    data        - path to csv, first row header, first column has labels
    numhidden   - number of hidden neurons

OUTPUT
    scores      - Tensor of predictions
*/

#include <torch/torch.h>
#include "lib/resnet.h"
#include <iostream>

int main() {
    auto dataset = torch::data::datasets::MNIST("./mnist")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));


    for (torch::data::Example<>& batch : *data_loader) {
        std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
        for (int64_t i = 0; i < batch.data.size(0); ++i) {
            std::cout << batch.target[i].item<int64_t>() << " ";
        }
        std::cout << std::endl;
    }

    torch::Device device("cpu");
    if (torch::cuda::is_available()){
        device = torch::Device("cuda:0");
    }

    Resnet<Block> resnet = resnet50();
    resnet.to(device);

    train(resnet);

}

void train(Resnet<Block> model) {
    torch::optim::Adam adam_optimizer(
        model->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
        
    for (int64_t epoch = 1; epoch <= 10; ++epoch) {
        int64_t batch_index = 0;
        for (torch::data::Example<>& batch : *data_loader) {
            // Train model with real images.
            model->zero_grad();
            torch::Tensor real_images = batch.data;
            torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
            torch::Tensor real_output = model->forward(real_images);
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();

            adam_optimizer.step();


            std::printf(
                "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
                epoch,
                kNumberOfEpochs,
                ++batch_index,
                batches_per_epoch,
                d_loss.item<float>(),
                g_loss.item<float>());
        }
    }
}