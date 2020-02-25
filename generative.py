from models.resnet import ResNet18
from models.simple_net import simpleNet
from ops import *
import torch
from initialize_model import load_data
import os
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import random

def fix_parameters(model):
    for param in model.parameters():
        param.requires_grad = False



def release_parameters(model):
    for param in model.parameters():
        param.requires_grad = True



def show_images(image_tensor):
    image = image_tensor.detach().cpu().numpy()
    image = image.transpose([0,2,3,1])
    fig = plt.figure(figsize=(28, 28))
    columns = 4
    rows = 5

    for i in range(1, columns * rows + 1):
        img = image[i].reshape([28,28])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()





def extract_features(trainset, sigma_list):
    '''

    :param images:
    :param targets:
    :return: the features and the corresponding sigma
    '''

    loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
    for batch_id, (images, _) in enumerate(loader):
        if batch_id == 300:
            break
        images = images.to(device)
        net.eval()
        _, feat = net(images)
        feat = feat.detach().cpu().numpy()
        if batch_id == 0:
            features = feat
        else:
            features = np.concatenate((features, feat), axis=0)

    label = np.array(trainset.targets[:30000])
    one_hot = np.zeros((label.size, label.max() + 1))
    one_hot[np.arange(label.size),label] = 1
    one_hot = torch.tensor(one_hot)
    one_hot = one_hot.to(device)

    sigma_vec = torch.mv(one_hot, sigma_list)

    return features, sigma_vec




def lr_scheduler(epoch, lr):
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




if __name__ == '__main__':

    fake_image_num = 300 #as large as possible
    epoch = 30
    init_lr = 0.00001

    batch_size = 32 # which is equal to the batch_size in the trainloader


    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    net = simpleNet()
    net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    trainset, testset, trainloader, testloader = load_data()
    _, h, w= trainset.data.shape
    channel = 1  #channel = 1 for mnist

    #initialize the fake images model and the sigma_list
    fake_images = init_images(fake_image_num, [h, w], channel)

    sigma_list = torch.ones(10, dtype=torch.float64)
    sigma_list = sigma_list.to(device)
    #load the model parameters
    state = torch.load('ckpt/init.ckpt')
    net.load_state_dict(state['state_dict'])

    for i in range(epoch):
        loss_image = 0
        loss_net = 0
        lr = lr_scheduler(epoch, init_lr)
        # shuffle the fake_images
        #shuffle_idx = torch.randperm(fake_image_num)
        #fake_images = fake_images[shuffle_idx]
        loss1 = torch.tensor(0)
        loss2 = torch.tensor(0)
        all_means, sigma_vec = extract_features(trainset, sigma_list)
        all_means = torch.tensor(all_means)
        all_means = all_means.to(device)


        for batch_id, (real_ims, targets) in enumerate(trainloader):
            num = real_ims.size()[0]

            indices = random.sample(list(range(fake_image_num)), num)
            fake_ims = fake_images[indices]
            #fake_ims = fake_images[batch_id * batch_size: batch_id * batch_size + num]
            fake_ims = torch.tensor(fake_ims, dtype=torch.float)

            fake_ims, real_ims = fake_ims.to(device), real_ims.to(device)


            ##########################################################
            #     update the network
            ################################################################
            if i % 2 == 1:

                fake_ims.requires_grad = False
                release_parameters(net)
                net.train()
                optimizer_net = torch.optim.Adam(net.parameters(), lr=0.000000001)  # todo adjust the learning rate
                optimizer_sigma = torch.optim.Adam([sigma_list], lr=0.001)
                _, real_feature = net(real_ims)
                _, fake_feature = net(fake_ims)

                loss1 = loss_for_network(real_feature, fake_feature, all_means, sigma_vec)
                loss1.backward()
                optimizer_net.step()






            #########################################################
            # update the images
            ##################################

            if i % 2 == 0:
                fake_ims.requires_grad = True
                fix_parameters(net)
                net.eval()

                optimizer_im = torch.optim.Adam([fake_ims], lr = 0.001)  # todo add the learning rate
                for j in range(2):
                    _, feature = net(fake_ims)
                    loss2 = loss_for_image(feature, all_means, sigma_vec)
                    loss2.backward()
                    optimizer_im.step()


                #fake_images[batch_id * batch_size: batch_id * batch_size + num] = fake_ims.detach().cpu().numpy()
                fake_images[indices] = fake_ims.detach().cpu().numpy()

            if batch_id % 100 == 0:
                print('epoch %d | batch: %d  | loss1: %.3f | loss2: %.3f'
                  % (i, batch_id, loss1 , loss2))
                show_images(fake_ims)
   




    np.save('fake_image.npy', fake_images)


    #save the model
    if not os.path.isdir('ckpt'):
        os.mkdir('ckpt')
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict()
    }
    torch.save(state, './ckpt/gen_net.ckpt')























