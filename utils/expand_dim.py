import torch
import torch.nn as nn


'''
Credit: https://github.com/joneswack/brats-pretraining/blob/master/jonas_net.py
'''

def conv2d_to_conv3d(layer):
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    new_weight = layer.weight.unsqueeze(2)
    kernel_size = tuple([1] + list(layer.kernel_size))
    stride = tuple([1] + list(layer.stride))
    padding = tuple([0] + list(layer.padding))

    new_layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    
    new_weight = layer.weight.unsqueeze(2)
    new_layer.weight.data = new_weight.data
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data
    
    return new_layer

def batch2d_to_batch3d(layer):
    num_features = layer.num_features
    eps = layer.eps
    momentum = layer.momentum
    affine = layer.affine
    
    new_bn = nn.BatchNorm3d(num_features, eps, momentum, affine)
    new_bn.load_state_dict(layer.state_dict())
    
    return new_bn

def pool2d_to_pool3d(layer):
    kernel_size = tuple([1, layer.kernel_size, layer.kernel_size])
    stride = tuple([1, layer.stride, layer.stride])
    padding = layer.padding
    dilation = layer.dilation
    return_indices = layer.return_indices
    ceil_mode = layer.ceil_mode
    
    return nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

def transform_module_list(module_list):
    transformed_list = []
    
    for item in module_list:
        if isinstance(item, nn.Conv2d): 
            transformed_list.append(conv2d_to_conv3d(item))
        elif isinstance(item, nn.BatchNorm2d):
            transformed_list.append(batch2d_to_batch3d(item))
        elif isinstance(item, nn.MaxPool2d):
            transformed_list.append(pool2d_to_pool3d(item))
        elif isinstance(item, nn.ReLU):
            transformed_list.append(item)
        elif isinstance(item, nn.Sequential):
            transformed_list += transform_module_list(item.children())
        else:
            transformed_list.append(item)
            
    return transformed_list

def translate_block(block, downsample=False):
    block.conv1 = conv2d_to_conv3d(block.conv1)
    block.bn1 = batch2d_to_batch3d(block.bn1)
    
    block.conv2 = conv2d_to_conv3d(block.conv2)
    block.bn2 = batch2d_to_batch3d(block.bn2)
    
    if block.downsample is not None:
        conv = conv2d_to_conv3d(block.downsample[0])
        bn = batch2d_to_batch3d(block.downsample[1])
        block.downsample = nn.Sequential(*[conv, bn])
    
    return block