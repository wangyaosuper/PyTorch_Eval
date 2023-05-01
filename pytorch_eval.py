
# -*- coding: utf-8 -*-

import torch
from torch import nn

import torchvision
from torchvision import transforms
from torch.utils import data

import os,getopt
import sys



def get_device(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# ------ AlexNet 模型创建 ----------

def init_weights(m):
  if type(m) == nn.Linear or type(m) == nn.Conv2d:
    nn.init.xavier_uniform_(m.weight)


def create_AlexNet():
  net = nn.Sequential(
    # 这⾥使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
    # 同时，步幅为4，以减少输出的⾼度和宽度。
    # 另外，输出通道的数⽬远⼤于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
    # 除了最后的卷积层，输出通道的数量进⼀步增加。
    # 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这⾥，全连接层的输出数量是LeNet中的好⼏倍。使⽤dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这⾥使⽤Fashion-MNIST，所以⽤类别数为10，⽽⾮论⽂中的1000
    nn.Linear(4096, 10))
  X = torch.randn(1, 1, 224, 224)
  for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
  net.apply(init_weights)
  return "AlexNet", net


class L2Pool(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(L2Pool, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        # 对输入图像的平方进行平均池化，并取平方根
        x = torch.sqrt(self.pool(x**2))
        return x

def create_AlexNet_With_L2Pool():
  net = nn.Sequential(
    # 这⾥使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
    # 同时，步幅为4，以减少输出的⾼度和宽度。
    # 另外，输出通道的数⽬远⼤于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
    # 除了最后的卷积层，输出通道的数量进⼀步增加。
    # 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    L2Pool(kernel_size=3, stride=2),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这⾥，全连接层的输出数量是LeNet中的好⼏倍。使⽤dropout层来减轻过拟合
    nn.Linear(1024, 6144), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(6144, 6144), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这⾥使⽤Fashion-MNIST，所以⽤类别数为10，⽽⾮论⽂中的1000
    nn.Linear(6144, 10))
  X = torch.randn(1, 1, 224, 224)
  for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
  net.apply(init_weights)
  return "AlexNet_With_L2Pool", net



# ------ NiN 模型创建 ----------
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

def create_NinNet():
  print("create NinNet")
  net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0,5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())
  X = torch.randn(1, 1, 224, 224)
  for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
  net.apply(init_weights)
  return "NinNet", net


# ------ 新增 GoogLeNet -----
class Inception(nn.Module):
  def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
    super(Inception, self).__init__(**kwargs)
    self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
    self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
    self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
    self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
    self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
    self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

  def forward(self, x):
    p1 = nn.functional.relu(self.p1_1(x))
    p2 = nn.functional.relu(self.p2_2(nn.functional.relu(self.p2_1(x))))
    p3 = nn.functional.relu(self.p3_2(nn.functional.relu(self.p3_1(x))))
    p4 = nn.functional.relu(self.p4_2(self.p4_1(x)))
    return torch.cat((p1, p2, p3, p4), dim=1)


def create_GoogLeNet():
  print("create GoogLeNet")
  b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

  b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

  b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
  
  b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
  
  b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten())
  
  net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

  X = torch.rand(size=(1, 1, 96, 96))
  for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

  net.apply(init_weights)
  return "GoogLeNet", net

# ----------- 新增ResNet ---------
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, 
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return "ResNet", nn.functional.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def create_ResNet():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                 nn.BatchNorm2d(64), nn.ReLU(), 
                                 nn.MaxPool2d(kernel_size=3, stride=2, 
                                              padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5, 
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    X = torch.rand(size=(1, 1, 96, 96))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

    net.apply(init_weights)
    return net

# ---------- 加载模型 ----------
def load_model(net, checkpoint):
    net.load_state_dict(checkpoint['model_state_dict'])
    model_name = checkpoint['model_name']
    epoch = checkpoint['epoch']
    test_acc = checkpoint['test_acc']
    train_acc = checkpoint['train_acc']
    loss = checkpoint['loss']
    print("Load model ", f": {model_name} , epoch={epoch}")
    print(f"Load this model with loss={loss:.3f}, test_acc={test_acc:.3f}, train_acc={train_acc:.3f}")
    return
 



# ----------- 数据加载 ----------

def get_dataloader_workers():
    """Use 4 processes to read the data.
    Defined in :numref:`sec_fashion_mnist`"""
    return 1

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.
    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))



# ----- lab -----

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    ret = float(reduce_sum(astype(cmp,y.dtype)))
    return ret


def evaluate_accuracy(net, data_iter, device):
    """Compute the accuracy for a model on a dataset using a GPU.
    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

  
"""  
def evaluate_accuracy(net, data_iter):
    if isinstance(net, nn.Module):
        net.eval()
        metric = Accumulator(2)
        with torch.no_grad():
            for X,y in data_iter:
                print("shape of input is ", X.shape)
                metric.add(accuracy(net(X),y), y.numel())
    return metric[0]/metric[1]
"""


def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hm:p:",["model=","path="])
    except getopt.GetoptError:
        print('pytorch_eval.py -m <model_checkpoint_file> -p <path_of_the_files_for_eval>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == ('-h'):
            print('pytorch_eval.py -m <model_checkpoint_file> -p <path_of_the_files_for_eval>')
            sys.exit()
        elif opt in ("-m", "--model"):
            model_file = arg
        elif opt in ("-p", "--path"):
            eval_data_path = arg

    print("in main")


    device = get_device()
    checkpoint = torch.load(model_file, map_location=device)
    model_name = checkpoint['model_name']

    train_iter, test_iter = load_data_fashion_mnist(batch_size=128, resize=224)

    model_switcher = {
        'AlexNet':create_AlexNet,
        'AlexNet_With_L2Pool':create_AlexNet_With_L2Pool,
        'NinNet':create_NinNet,
        'GoogLeNet':create_GoogLeNet,
        'ResNet':create_ResNet}

    if (not model_name in model_switcher):
        print(f"Error !!! Model:{model_name} is not supported")
        sys.exit(2)

    model_name, net = model_switcher.get(model_name)()
    load_model(net, checkpoint)
    net.eval()

    acc = evaluate_accuracy(net, test_iter, device)
    print(f"Accuracy: {acc:.3f}")
    

    print("out main")



if __name__ == "__main__":
    print("call main")
    main(sys.argv[1:])
    print("main finished")
