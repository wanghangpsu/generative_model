def update_image(fake_images, kernel_means, sigma_vec):

    '''
    :param fake_images:
    :param kernel_means: we use all the features of the real image as the kernel means
    :param sigma_vec:
    :return: nothing, fake_images are updated
    '''

    fake_images.to(device)
    fake_images.requires_grad = True
    fix_parameters(net)
    net.eval()


    optimizer_im = torch.optim.Adam(fake_images)  #todo add the learning rate
    feature, _ = net(fake_images)
    loss = loss_for_image(feature, kernel_means, sigma_vec)
    loss.backward()
    optimizer_im.step()

    return loss.item()

def update_net(real_images, fake_images, kernel_means, sigma_vec):
    fake_images.requires_grad = False
    release_parameters(net)
    net.train()
    optimizer_net = torch.optim.Adam(net.parameters, lr=0.000001) #todo adjust the learning rate
    real_feature, _ = net(real_images)
    fake_feature, _ = net(fake_images)
    loss = loss_for_network(real_feature, fake_feature, kernel_means, sigma_vec)
    loss.backward()
    optimizer_net.step()
    return loss