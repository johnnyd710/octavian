import torch
import click
import time
from torchvision import datasets, transforms, models
from torchvision.models.resnet import ResNet, BasicBlock
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# helper functions
def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    try:
        return metric_fn(true_y, pred_y, average="macro")
    except:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def loss_function(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.cross_entropy(output, labels)

class AmdResnet(ResNet):
  input_channels: int = 1
  def __init__(self):
    super(AmdResnet, self).__init__(
      BasicBlock,
      [2,2,2,2],
      num_classes=2
    )
    self.conv1 = torch.nn.Conv2d(self.input_channels, 64,
      kernel_size=(7,7),
      stride=(2,2),
      padding=(3,3), bias=False)

@click.command()
@click.option('--restore', default=1, help='restore from checkpoint?')
@click.option('--epochs', default=10, help='restore from checkpoint?')
def run(restore, epochs):
  """ loads data from mnist for testing resnet """
  transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
  trainset = datasets.ImageFolder('./data/iChallene-AMD-Training400/Training400', transform=transform)
  # trainset = datasets.MNIST('./mnist-py', download=True, transform=transform)
  # testset = datasets.MNIST('./mnist-py', download=True, transform=transform, train=False)
  testset = trainset
  loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
  val_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)

  # set device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # configure resnet
  model = AmdResnet()

  # configure optimizer
  optimizer = torch.optim.Adam(model.parameters(), 2e-4)
  
  if restore:
    model.load_state_dict(torch.load('./checkpoint.pt'))
    optimizer.load_state_dict(torch.load('./optimizer-checkpoint.pt'))

  losses = []
  batches = len(loader)
  val_batches = len(val_loader)

  for epoch in range(epochs):
    total_loss = 0
    for i, batch in enumerate(loader):
      model.zero_grad()
      images = batch[0].to(device)
      labels = batch[1].to(device)
      output = model.forward(images)

      loss = loss_function(output, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss

      print("Batch: {:d} Loss: {:.4f}".format(i, total_loss/(i+1)))
      torch.save(model.state_dict(), './checkpoint.pt')
      torch.save(optimizer.state_dict(), './optimizer-checkpoint.pt')
    
    print("Epoch {:d} complete out of {:d}".format(epoch+1, epochs))
    print("======================================")

  # ----------------- VALIDATION  ----------------- 
  val_losses = 0
  precision, recall, f1, accuracy = [], [], [], []
  
  # set model to evaluating (testing)
  model.eval()
  with torch.no_grad():
      for i, data in enumerate(val_loader):
          X, y = data[0].to(device), data[1].to(device)

          # plt.imshow(X[0].permute(1, 2, 0))
          # plt.show()

          # exit()

          outputs = model(X) # this get's the prediction from the network

          val_losses += loss_function(outputs, y)

          predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
          
          # calculate P/R/F1/A metrics for batch
          for acc, metric in zip((precision, recall, f1, accuracy), 
                                  (precision_score, recall_score, f1_score, accuracy_score)):
              acc.append(
                  calculate_metric(metric, y.cpu(), predicted_classes.cpu())
              )
        
  print(f"validation loss: {val_losses/val_batches}")
  print_scores(precision, recall, f1, accuracy, val_batches)
  # losses.append(total_loss/batches) # for plotting learning curve

if __name__ == "__main__":
  start_ts = time.time()
  run()
  print(f"Training time: {time.time()-start_ts}s")
