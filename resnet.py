import torch
import click
from torchvision import datasets, transforms

class Block(torch.nn.Module):
  
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    stride = 1,
    identity_downsample = torch.nn.Sequential(),
    expansion = 4
  ):
    super(Block, self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0)
    self.conv2 = torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1)
    self.conv3 = torch.nn.Conv2d(in_channels, out_channels * expansion, 1, 1, 0)
    self.bn1 = torch.nn.BatchNorm2d(out_channels)
    self.bn2 = torch.nn.BatchNorm2d(out_channels)
    self.bn3 = torch.nn.BatchNorm2d(out_channels*expansion)
    self.identity_downsample = identity_downsample
    self.relu = torch.nn.ReLU()
    self.expansion = expansion

  def forward(self, x):
    assert(torch.is_tensor(x))
    identity = x
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.bn3(self.conv3(x))

    if (self.identity_downsample != None):
      identity = self.identity_downsample.forward(identity)

    x += identity
    x = self.relu(x)
    return x

class ResNet(torch.nn.Module):
  def __init__(
    self,
    layers: list,
    image_channels: int,
    num_class: int
  ):
    super(ResNet, self).__init__()
    self.in_channels = 64
    self.expansion = 4
    self.conv1 = torch.nn.Conv2d(image_channels, self.in_channels, 7, 2, 3)
    self.bn1 = torch.nn.BatchNorm2d(self.in_channels)
    self.layer1 = self.make_layer(64, layers[0])
    self.layer2 = self.make_layer(128, layers[1])
    self.layer3 = self.make_layer(256, layers[2])
    self.layer4 = self.make_layer(512, layers[3])
    self.avg_pool2d = torch.nn.AvgPool2d(7, 1)
    self.fc = torch.nn.Linear(512 * self.expansion, num_class)

  def make_layer(
    self,
    planes: int,
    blocks: int,
    stride = 1
  ) -> torch.nn.Sequential:
    if (stride != 1 or self.in_channels != planes * self.expansion):
      downsample = torch.nn.Sequential(
        torch.nn.Conv2d(self.in_channels, planes * self.expansion, 1, stride),
        torch.nn.BatchNorm2d(planes * self.expansion)
      )

    layers = []
    layers.append(Block(self.in_channels, planes, stride, downsample))
    in_channels = planes * self.expansion;
    for i in range(0, blocks):
      layers.append(Block(in_channels, planes))

    return torch.nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv1.forward(x)
    x = self.bn1.forward(x)
    x = torch.relu(x)
    x = torch.max_pool2d(x, 3, 2, 1)
    x = self.layer1.forward(x)
    x = self.layer2.forward(x)
    x = self.layer3.forward(x)
    x = self.layer4.forward(x)
    # dont know about this part...
    # x = self.avg_pool2d.forward(x)
    # x = x.view
    # end
    x = self.fc.forward(x)
    return x

@click.command()
@click.option('--restore', default=1, help='restore from checkpoint?')
@click.option('--epochs', default=10, help='restore from checkpoint?')
def load_data(restore, epochs):
  """ loads data from mnist for testing resnet """
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
  trainset = datasets.MNIST('./mnist', download=True, transform=transform)
  loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

  # configure resnet
  resnet = ResNet([3,4,6,3], 1, 10)

  # configure optimizer
  optimizer = torch.optim.Adam(resnet.parameters(), 2e-4)
  
  if restore:
    resnet.load_state_dict(torch.load('./checkpoint.pt'))
    optimizer.load_state_dict(torch.load('./optimizer-checkpoint.pt'))

  for epoch in range(epochs):
    for i, batch in enumerate(loader):
      resnet.zero_grad()
      images = batch[0]
      labels = batch[1]
      output = resnet.forward(images)

      print(output)

      loss = torch.nn.functional.cross_entropy(output, labels)

if __name__ == "__main__":
  load_data()
