from models.resnet import ResNet18
from ops import *
import torch
from initialize_model import load_data
import os
import torch.backends.cudnn as cudnn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18()
net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

trainset, testset, trainloader, testloader = load_data()

state = torch.load('ckpt/init.ckpt')
net.load_state_dict(state['state_dict'])

list = []
num_class = 10

for i in range(num_class):

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        indices = [j for j, x in enumerate(targets) if x == i]
        inputs = inputs[indices]
        if indices == []:
            continue
        inputs = inputs.to(device)
        _, feature = net(inputs)
        feature = feature.detach().cpu().numpy()
        if batch_idx == 0:
            features = feature
        else:
            features = np.concatenate((features, feature), axis=0)
        print(feature)
        mean = np.mean(features, axis=0, keepdims=True)
        b = (features - mean) ** 2
        sigma = np.sum(b)





