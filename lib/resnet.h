/**
 * @author john dimatteo
 * 
**/

#include <torch/torch.h>

using Convolution = torch::nn::Conv2d;
using BatchNorm = torch::nn::BatchNorm;
using Tensor = torch::Tensor;
using Sequential = torch::nn::Sequential;
using Relu = torch::nn::ReLU;


torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kernel_size,
                                      int64_t stride=1, int64_t padding=0, bool with_bias=false){
  torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size)
    .stride(stride)
    .padding(padding)
    .bias(with_bias);
  return conv_options;
}

struct Block : torch::nn::Module {

    Convolution conv1;
    BatchNorm bn1;
    Convolution conv2;
    BatchNorm bn2;
    Convolution conv3;
    BatchNorm bn3;
    torch::nn::Sequential identity_downsample;
    Relu relu{ Relu() };
    int expansion{4};

    Block(int in_channels,
        int out_channels,
        Sequential identity_downsample = Sequential(),
        int stride = 1 ) :
    conv1(conv_options(in_channels, out_channels, /* kernel */ 1, /* stride */ 1, /* padding */ 0)),
    bn1(out_channels),
    conv2(conv_options(out_channels, out_channels, /* kernel */ 3, stride, /* padding */ 1)),
    bn2(out_channels),
    conv3(conv_options(out_channels, out_channels*expansion, /* kernel */ 1, 1, /* padding */ 0)),
    bn3(out_channels*expansion),
    identity_downsample(identity_downsample)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        if (!identity_downsample->is_empty()) {
            register_module("identity_downsample", identity_downsample);
        }
    }

    Tensor forward(Tensor x) {
        Tensor identity{x.clone()};
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = relu->forward(x);
        x = conv2->forward(x);
        x = bn2->forward(x);
        x = relu->forward(x);
        x = conv3->forward(x);
        x = bn3->forward(x);

        if (!identity_downsample->is_empty()){
            identity = identity_downsample->forward(identity);
        }

        x += identity;
        x = relu->forward(x);
        return x;
    }
};
