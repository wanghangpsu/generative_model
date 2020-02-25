

import numpy as np
import torch



def gaussian(x, mean, var):
    '''
    x: torch tensor a matrix with size (sample_size, feature_dimension)
    mean: torch tensor
    var: torch tensor
    return the pdf of the guassian distribution with mean and var
    '''
    k = x.size()[1]
    a = x - mean
    b = torch.mm(a, torch.inverse(var))
    b = torch.mm(b, torch.transpose(a, 0, 1))
    det_sigma = np.linalg.det(var.detach().cpu.numpy())
    return torch.exp(-1 * b) / np.sqrt((2 * np.pi) ** k * det_sigma)

def gaussian_mixture(x, prior, mean_list, var_list):
    '''
    x: tensor
    prior: a list of prior of different component
    mean_list: a list of mean(tensor)
    var_list: list of covariance matrix(tensor)
    '''

    num_sample = x.size()[0]
    p = torch.zeros([num_sample, 1])

    for i in range(len(prior)):
        p += prior[i] * gaussian(x, mean_list[i], var_list[i])

    return p


def init_images(sample_size, image_size, dimension):
    h, w = image_size
    return np.random.rand(sample_size, h, w, dimension)





def compute_gaussian_param(num_component, dataset, net):
    '''
    compute the mu and sigma in the Gaussian Mixture Model based on the sample mean and Covariance
    num_component: int, number of components(i.e. number of classes)
    dataset: torch.dataset object, real images and label
    net:

    return mu, Sigma  (lists of mu and sigma) both lists of torch.tensor
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean_list = []
    var_list = []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=2)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        _, penul_feature = net(inputs)
        penul_feature = penul_feature.detach().cpu().numpy()
        if batch_idx == 0:
            features = penul_feature
        else:
            features = np.concatenate((features, penul_feature), axis=0)

    features = np.asarray(features)
    for i in range(num_component):
        indices = [j for j, x in enumerate(dataset.targets) if x == i]

        features_i = features[indices]
        mean_list.append(torch.tensor(np.mean(features_i, axis=0, keepdims=True)))
        var_list.append(torch.tensor(np.var(features_i, axis=0)))

    return mean_list, var_list



