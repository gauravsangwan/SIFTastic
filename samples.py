import torch
import torch.nn as nn
import data_loader
import numpy as np
import os
import models
import lib.adversary as adversary

from torchvision import transforms
from torch.autograd import Variable

from config import *


def main():
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)

    if MODEL == 'resnet':
        model = models.ResNet34(NUMBER_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH))
        in_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        min_pixel = -2.42906570435
        max_pixel = 2.75373125076

        if DATASET == 'cifar10':
            if ADVERSARY == 'fgsm':
                random_noise_size = 0.25/4
            elif ADVERSARY == 'bim':
                random_noise_size = 0.13/2
            elif ADVERSARY == 'deepfool':
                random_noise_size = 0.25/4
            elif ADVERSARY == 'cw':
                random_noise_size = 0.05/2
        elif DATASET == 'cifar100':
            if ADVERSARY == 'fgsm':
                random_noise_size = 0.25/8
            elif ADVERSARY == 'bim':
                random_noise_size = 0.13/4
            elif ADVERSARY == 'deepfool':
                random_noise_size = 0.13/4
            elif ADVERSARY == 'cw':
                random_noise_size = 0.05/2
        else:
            if ADVERSARY == 'fgsm':
                random_noise_size = 0.25/4
            elif ADVERSARY == 'bim':
                random_noise_size = 0.13/2
            elif ADVERSARY == 'deepfool':
                random_noise_size = 0.126
            elif ADVERSARY == 'cw':
                random_noise_size = 0.05/1
    elif MODEL == 'vit':
        model = models.vit224(NUMBER_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH))
        in_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        min_pixel = -1.98888885975
        max_pixel = 2.12560367584

        if DATASET == 'cifar10':
            if ADVERSARY == 'fgsm':
                random_noise_size = 0.21/4
            elif ADVERSARY == 'bim':
                random_noise_size = 0.21/4
            elif ADVERSARY == 'deepfool':
                random_noise_size = 0.26/10
            elif ADVERSARY == 'cw':
                random_noise_size = 0.03/2
        elif DATASET == 'cifar100':
            if ADVERSARY == 'fgsm':
                random_noise_size = 0.21/8
            elif ADVERSARY == 'bim':
                random_noise_size = 0.21/8
            elif ADVERSARY == 'deepfool':
                random_noise_size = 0.26/8
            elif ADVERSARY == 'cw':
                random_noise_size = 0.06/5
        else:
            if ADVERSARY == 'fgsm':
                random_noise_size = 0.21/4
            elif ADVERSARY == 'bim':
                random_noise_size = 0.21/4
            elif ADVERSARY == 'deepfool':
                random_noise_size = 0.32/5
            elif ADVERSARY == 'cw':
                random_noise_size = 0.07/2
    else:
        raise ValueError('Model not supported')
    
    model = model.to(DEVICE)
    print('Loading Model ',MODEL,'!!!!')

    print('Loading target data', DATASET, '!!!!')

    _, test_loader = data_loader.getTargetDataSet(DATASET, BATCH_SIZE, in_transform,DATAROOT)
    
    print('Starting Adversarial Attack')
    print('Adversary: ', ADVERSARY, 'DATASET: ',DATASET)
    
    model.eval()

    adv_data_tot, clean_data_tot, noise_data_tot = 0,0,0
    label_tot = 0

    correct, adv_correct, noise_correct = 0,0,0
    total, generated_noise = 0,0

    criterion = nn.CrossEntropyLoss().cuda()

    selected_list = []
    selected_index = 0

    '''
    Overall flow of the code in this file:
    - start the attack on dataset iterate over the test loader
    - compute the accuracy on clean data
    - generate adversarial images
    - initialize the attack and get adv_data
    - measure the noise 
    - compute the accuracy on adversarial data
    - Save the clean data total, adv data total, noise total and label total
    '''
    for data,target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data,volatile=True), Variable(target)
        #print(data.shape)
        output = model(data)
        pred = output.data.max(1)[1]
        #print(output.size())
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()

        noisy_data = torch.add(data.data, random_noise_size, torch.randn(data.size()).cuda())
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)

        #generate adversarial 
        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        if ADVERSARY == 'fgsm':
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float()-0.5)*2
            if MODEL == 'vit':
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
            else: #if resnnet
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        elif ADVERSARY == 'bim':
            gradient = torch.sign(inputs.grad.data)
            for k in range(5):
                inputs = torch.add(inputs.data, EPSILON, gradient)
                inputs = torch.clamp(inputs, min_pixel, max_pixel)
                inputs = Variable(inputs, requires_grad=True)
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                gradient = torch.sign(inputs.grad.data)
                if MODEL == 'vit':
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
                else:
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        
        if ADVERSARY == 'deepfool':
            _, adv_data = adversary.deepfool(model, data.data.clone(), target.data.cpu(), NUMBER_CLASSES, step_size=EPSILON, train_mode = False)
            adv_data = adv_data.cuda()
        elif ADVERSARY == 'cw':
            _, adv_data = adversary.cw(model, data.data.clone(), target.data.cpu(), 1.0, 'l2', crop_frac=1.0)
        else :
            adv_data = torch.add(inputs.data,EPSILON, gradient)
        
        adv_data = torch.clamp(adv_data, min_pixel, max_pixel)

        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)

        if total == 0:
            flag = 1
            adv_data_tot = adv_data.clone().cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)

        # Clean Accuracy
        output = model(Variable(adv_data, volatile=True))
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()
            

        # Noisy Accuracy
        output = model(Variable(noisy_data, volatile=True))
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()

        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
            selected_index += 1
        
        total += data.size(0)
    

    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    torch.save(clean_data_tot, '%s/clean_data_%s_%s_%s.pth' % (OUTF, MODEL, DATASET, ADVERSARY))
    torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % (OUTF, MODEL, DATASET, ADVERSARY))
    torch.save(noisy_data_tot, '%s/noisy_data_%s_%s_%s.pth' % (OUTF, MODEL, DATASET, ADVERSARY))
    torch.save(label_tot, '%s/label_%s_%s_%s.pth' % (OUTF, MODEL, DATASET, ADVERSARY))
    
    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
    print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))    



if __name__ == '__main__':
    main()
