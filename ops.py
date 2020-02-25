import numpy as np
import torch


def kernel_density(z, features, sigma_vec):
    '''
    function of computing the kernel density p(z)
    sigma_vec: assign different sigma to different mean according to the label of the mean(real data point)
    '''
    l2_norms = torch.sum((z - features) ** 2, dim=1)
    density = torch.exp(l2_norms * -1.0 / (2 * sigma_vec ** 2)) /(sigma_vec * np.sqrt(2 * np.pi))
    density = torch.mean(density)
    print(density)
    return density


def loss_for_network(real_feature, fake_feature, all_means, sigma_vec):

    loss = 0
    for i in range(real_feature.size()[0]):
        loss += kernel_density(real_feature[i], all_means, sigma_vec)

    for j in range(fake_feature.size()[0]):
        loss -= kernel_density(fake_feature[j], all_means, sigma_vec)
    return loss

def loss_for_image(fake_feature, all_means, sigma_vec):

    loss = 0
    for j in range(fake_feature.size()[0]):
        loss -= torch.log(kernel_density(fake_feature[j], all_means, sigma_vec))
    return loss

def get_sigma_vec(sigma_list, lable):
    '''
    :param sigmasq_list: list of torch.tensor
    :param lable:  list of interger
    :return: sigmasq_vec
    '''
    sigma_vec = [sigma_list[x] for x in lable]
    return torch.tensor(sigma_vec)


def init_images(sample_size, image_size, dimension):
    h, w = image_size
    return np.random.rand(sample_size, dimension, h, w)

