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
  torch::Device device("cpu");
  if (torch::cuda::is_available()){
    device = torch::Device("cuda:0");
  }

  torch::Tensor t = torch::rand({2, 3, 224, 224}).to(device);
  std::cout << t.sizes() << std::endl;
  Resnet<Block> resnet = resnet50();
  resnet.to(device);

  t = resnet.forward(t);
  std::cout << t.sizes() << std::endl;
  std::cout << t << std::endl;
}