from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch
from models.resnet import ResNet18
from models.simple_net import simpleNet
import torch.backends.cudnn as cudnn
import os
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_scheduler(epoch))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 0:
            print('epoch %d | Loss: %.3f | trainAcc: %.3f%%'
                  % (epoch, train_loss / (batch_idx + 1), 100. * correct / total))
    return net


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('epoch %d  Loss: %.3f | testAcc: %.3f%% (%d/%d)'
                % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def lr_scheduler(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)

    return lr


def load_data(data='mnist'):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if data == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    elif data == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainset, testset, trainloader, testloader


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('initialize model')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--dataset', choices=['cifar10', 'mnist'], default='mnist',
                                help='choose the dataset you want to use')
    model_settings.add_argument('--lr', type=float, default=0.001,
                                help='learning rate')

    model_settings.add_argument('--step', type=int, default=200,
                                help='# of epochs')

    return parser.parse_args()



if __name__ == '__main__' :
    args = parse_args()

    step = args.step
    best_acc = 0
    net = simpleNet()

    criterion = nn.CrossEntropyLoss()

    trainset, testset, trainloader, testloader = load_data(data=args.dataset)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if os.path.exists('ckpt/init.ckpt'):
        state = torch.load('ckpt/init.ckpt')
        net.load_state_dict(state['state_dict'])
        start_epoch = state['epoch']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + step):
        net = train(epoch)
        test(epoch)

        # Save model
        if not os.path.isdir('ckpt'):
            os.mkdir('ckpt')

        state = {
            'epoch': start_epoch + step,
            'state_dict': net.state_dict()
        }

        torch.save(state, './ckpt/init.ckpt')






