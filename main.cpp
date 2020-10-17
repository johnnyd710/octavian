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

int main(int argc, char **argv) {
  // Check the number of parameters
  if (argc < 2) {
    // Tell the user how to run the program
    std::cerr << "Usage: " << argv[0] << " RESTORE_FROM_CHECKPOINTS NAME EPOCHS" << std::endl;
    return 1;
  }
  
  int restoreFromCheckpoint{atoi(argv[1])};
  std::string name{argv[2]};
  int kBatchSize{10};
  int kNumberOfEpochs{atoi(argv[3])};
  int kCheckpointEvery{100};
  int checkpoint_counter{0};

  auto dataset = torch::data::datasets::MNIST("./mnist")
      .map(torch::data::transforms::Normalize<>(0.5, 0.5))
      .map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));


  torch::Device device("cpu");
  if (torch::cuda::is_available()){
      device = torch::Device("cuda:0");
  }

  std::vector<int> layers{3, 4, 6, 3};
  Resnet resnet(layers, 1 /* nummber of channels */, 10 /* number of classes*/);
  resnet->to(device);

  torch::optim::Adam adam_optimizer(resnet->parameters(), torch::optim::AdamOptions(2e-4));

  if (restoreFromCheckpoint == 1) {
    std::cout << "restore from checkpoint" << std::endl;
    torch::load(resnet, "checkpoint.pt");
    torch::load(adam_optimizer, "optimizer-checkpoint.pt");
  }

  // for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
  //   int64_t batch_index = 0;
  //   for (torch::data::Example<>& batch : *data_loader) {
  //     // Train model with real images.
  //     resnet->zero_grad();
  //     torch::Tensor real_images = batch.data;
  //     torch::Tensor real_labels = batch.target;
  //     torch::Tensor real_output = resnet->forward(real_images);

  //   //   std::cout << real_output << real_labels << std::endl;
      
  //     torch::Tensor d_loss_real = torch::nn::functional::cross_entropy(real_output, real_labels);
  //     d_loss_real.backward();

  //     adam_optimizer.step();


  //     std::printf("\r[%2ld/%2ld] batch # [%3ld] (x10 images per batch) || loss: %.4f",
  //     epoch,
  //     kNumberOfEpochs,
  //     ++batch_index,
  //     d_loss_real.item<float>());

  //     if (batch_index % kCheckpointEvery == 0) {
  //       // Checkpoint the model and optimizer state.
  //       torch::save(resnet, "checkpoint.pt");
  //       torch::save(adam_optimizer, "optimizer-checkpoint.pt");
  //       std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
  //     }
  //   }
  // }

  std::cout << "Completed training" << std::endl;

  torch::save(resnet, name);

  auto dataset2 = torch::data::datasets::MNIST("./mnist", torch::data::datasets::MNIST::Mode::kTrain)
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

  auto data_loader2 = torch::data::make_data_loader(
    std::move(dataset2),
    torch::data::DataLoaderOptions().workers(2));

  resnet->eval();

  int64_t batch_index = 0;
  for (torch::data::Example<>& image : *data_loader2) {
    resnet->zero_grad();
    torch::Tensor prediction = resnet->forward(image.data);
    torch::Tensor labels = image.target;
    std::cout << "Prediction: " << prediction << " Labels: " << labels << std::endl;
    std::cout << "Loss: " << torch::nn::functional::cross_entropy(prediction, labels) << std::endl;
    if (batch_index++ > 20) {
      break;
    }
  }
}

// something is wrong with model loading