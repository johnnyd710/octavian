/**
 * @author john dimatteo
 * 
**/

#include <torch/torch.h>

using Convolution = torch::nn::Conv2d;
using BatchNorm = torch::nn::BatchNorm2d;
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

    Convolution conv1, conv2, conv3;
    BatchNorm bn1, bn2, bn3;
    torch::nn::Sequential identity_downsample;
    Relu relu{ Relu() };
    static const int expansion{4};

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
        x = relu(bn1(conv1(x)));
        x = relu(bn2(conv2(x)));
        x = bn3(conv3(x));

        if (!identity_downsample->is_empty()){
            identity = identity_downsample->forward(identity);
        }

        x += identity;
        x = relu(x);
        return x;
    }
};

template <class Block>
struct Resnet : torch::nn::Module {

    int in_channels{64};
    Convolution conv1;
    BatchNorm bn1;
    torch::nn::Sequential layer1, layer2, layer3, layer4;
    Relu relu = Relu();
    torch::nn::Linear fc;

    Resnet(std::vector<int> layers, int image_channels, int num_classes) :
        conv1(conv_options(image_channels, in_channels, /* kernel */ 7, /* stride */ 2, /* padding */ 3)),
        bn1(in_channels),
        layer1(_make_layer(64, layers[0])),
        layer2(_make_layer(128, layers[1], 2)),
        layer3(_make_layer(256, layers[2], 2)),
        layer4(_make_layer(512, layers[3], 2)),
        fc(512 * Block::expansion, num_classes)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
    }

    Tensor forward(Tensor x) {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 3, 2, 1);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = torch::avg_pool2d(x, 7, 1);
        x = x.view({x.sizes()[0], -1});
        x = fc->forward(x);

        return x;
    }


    private:
        torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1) {
            torch::nn::Sequential downsample;
            if (stride != 1 or in_channels != planes * Block::expansion){
                downsample = torch::nn::Sequential(
                    Convolution(conv_options(in_channels, planes * Block::expansion, 1, stride)),
                    BatchNorm(planes * Block::expansion)
                );
            }
            torch::nn::Sequential layers;
            layers->push_back(Block(in_channels, planes, downsample, stride));
            in_channels = planes * Block::expansion;
            for (int64_t i = 0; i < blocks; i++){
                layers->push_back(Block(in_channels, planes));
            }

            return layers;
        }
};

Resnet<Block> resnet50(){
  Resnet<Block> model({3, 4, 6, 3}, 3, 2);
  return model;
}