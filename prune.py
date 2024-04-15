import torch
import torch.nn as nn
import numpy as np
from config import *
from reconv import *
from resnet_cifar import *
from compute_flops import *

# Prune the network
def prune(network, prune_layers, prune_channels, prune_eigvs = None, prune_optvs = None, magnitude=False, random=False, hybrid=False, independent_prune_flag=False, opt=False, rev=False):
    count = 0 # count for indexing 'prune_channels'
    conv_count = 1 # conv count for 'indexing_prune_layers'
    dim = 0 # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None # residue is need to prune by 'independent strategy'
    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            if dim == 1:
                new_, residue = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1

            if 'conv%d'%conv_count in prune_layers:
                if magnitude:
                    channel_index = get_channel_index_magnitude(network.features[i].weight.data, prune_channels[count], conv_count, residue, rev)
                elif hybrid:
                    channel_index = get_channel_index_hybrid(network.features[i].weight.data, prune_channels[count], conv_count, prune_eigvs[count], residue)
                elif random:
                    channel_index = get_channel_index_random(network.features[i].weight.data, prune_channels[count], conv_count, residue)
                else:
                    channel_index = get_channel_index(network.features[i].weight.data, prune_channels[count], conv_count, prune_eigvs[count], residue, rev)
                new_ = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1
                count += 1
            else:
                residue = None
            conv_count += 1

        elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(network.features[i], channel_index)
            network.features[i] = new_

    # update to check last conv layer pruned
    if 'conv13' in prune_layers:
        network.classifier[0] = get_new_linear(network.classifier[0], channel_index)
    
    return network

def prune_resnet56(args, model, skip, prune_prob, eigvs_dict, rev=False):
    layer_id = 1
    cfg = []
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in skip:
                cfg_mask.append(torch.ones(out_channels))
                cfg.append(out_channels)
                layer_id += 1
                continue
            if layer_id % 2 == 0:
                if layer_id <= 18:
                    stage = 0
                elif layer_id <= 36:
                    stage = 1
                else:
                    stage = 2
                prune_prob_stage = prune_prob[stage]
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                kernel = m.weight.clone()
                
                scores = eigvs_dict[layer_id]
#                 print(scores)
#                 L1_norm = np.sum(weight_copy, axis=(1,2,3))

                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(scores)
#                 print(arg_max)
                # For ablation studies.
                # Keep the lowest instead
                if rev:
                    arg_max_rev = arg_max[:num_keep]
                else:
                    arg_max_rev = arg_max[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                cfg.append(num_keep)
                layer_id += 1
                continue
            layer_id += 1
    
    import re
    PAT = re.compile("\d+")
    depth = int(PAT.search(args.model).group(0))
    if args.data_set == 'CIFAR10':
        dataset = 'cifar10'
    elif args.data_set == 'CIFAR100':
        dataset = 'cifar100'
    else:
        print("Unsupported dataset")
    newmodel = resnet(dataset=dataset, depth=depth, cfg=cfg)

    # Copy the newmask
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    conv_count = 1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if conv_count % 2 == 0:
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                conv_count += 1
                continue
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

        num_parameters = sum([param.nelement() for param in newmodel.parameters()])
        model = newmodel

    print("number of parameters: "+str(num_parameters))
    
    print_model_param_flops(model.cpu(), input_res=32)
    print_model_param_nums(model.cpu())
    
    model.cuda()
    return model

def prune_resnet110(args, model, skip, prune_prob, eigvs_dict):
    layer_id = 1
    cfg = []
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in skip:
                cfg_mask.append(torch.ones(out_channels))
                cfg.append(out_channels)
                layer_id += 1
                continue
            if layer_id % 2 == 0:
                stage = layer_id // 36
                if layer_id <= 36:
                    stage = 0
                elif layer_id <= 72:
                    stage = 1
                elif layer_id <= 108:
                    stage = 2
                prune_prob_stage = prune_prob[stage]
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                kernel = m.weight.clone()
                
                scores = eigvs_dict[layer_id]
                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(scores)
                arg_max_rev = arg_max[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                cfg.append(num_keep)
                layer_id += 1
                continue
            layer_id += 1
    
    import re
    PAT = re.compile("\d+")
    depth = int(PAT.search(args.model).group(0))
    if args.data_set == 'CIFAR10':
        dataset = 'cifar10'
    elif args.data_set == 'CIFAR100':
        dataset = 'cifar100'
    else:
        print("Unsupported dataset")
    newmodel = resnet(dataset=dataset, depth=depth, cfg=cfg)

    # Copy the newmask
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    conv_count = 1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if conv_count % 2 == 0:
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                conv_count += 1
                continue
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
        num_parameters = sum([param.nelement() for param in newmodel.parameters()])
        model = newmodel

    print("number of parameters: "+str(num_parameters))
    
    print_model_param_flops(model.cpu(), input_res=32)
    print_model_param_nums(model.cpu())
    
    model.cuda()
    return model

def prune_resnet34(args, model, skip, prune_prob, eigvs_dict):
    from resnet_imagenet import resnet34
    layer_id = 1
    cfg = []
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1,1):
                continue
            out_channels = m.weight.data.shape[0]
            if layer_id in skip:
                cfg_mask.append(torch.ones(out_channels))
                cfg.append(out_channels)
                layer_id += 1
                continue
            if layer_id % 2 == 0:
                if layer_id <= 6:
                    stage = 0
                elif layer_id <= 14:
                    stage = 1
                elif layer_id <= 26:
                    stage = 2
                else:
                    stage = 3
                prune_prob_stage = prune_prob[stage]
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                
                scores = eigvs_dict[layer_id]

                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(scores)
                arg_max_rev = arg_max[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                cfg.append(num_keep)
                layer_id += 1
                continue
            layer_id += 1

    assert len(cfg) == 16, "Length of cfg variable is not correct."

    newmodel = resnet34(cfg=cfg)

    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    conv_count = 1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            if m0.kernel_size == (1,1):
                # Cases for down-sampling convolution.
                m1.weight.data = m0.weight.data.clone()
                continue
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if conv_count % 2 == 0:
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                conv_count += 1
                continue
        elif isinstance(m0, nn.BatchNorm2d):
            assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    return newmodel, cfg

def get_channel_index(kernel, num_elimination, conv_ct, eigvs=None, residue=None, rev=False):
    # get cadidate channel index for pruning
    ## 'residue' is needed for pruning by 'independent strategy'

#     sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    print(f"Conv Layer {conv_ct}:")
#     print("Doing Eigen Analysis")
    if eigvs is None:
        eigvs = calculate_eigvs(kernel, FM_SIZES[conv_ct])
    eigvs = torch.tensor(eigvs)
    if rev:
        vals, args = torch.sort(eigvs, descending=True)
    else:
        vals, args = torch.sort(eigvs)
#     print(vals[:num_elimination])
#     print("Done")
    return args[:num_elimination].tolist()

def get_channel_index_magnitude(kernel, num_elimination, conv_ct, residue=None, rev=False):
    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    
    if rev:
        vals, args = torch.sort(sum_of_kernel, descending=True)
    else:
        vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()
    
def get_channel_index_random(kernel, num_elimination, conv_ct, residue=None):
    import random
    channels = random.sample(range(0, NUM_CHANNELS[conv_ct-1]), num_elimination)
    return channels

def get_channel_index_hybrid(kernel, num_elimination, conv_ct, eigvs=None, residue=None):
    print(f"Conv Layer {conv_ct}:")
#     print("Doing Eigen Analysis")
    if eigvs is None:
        eigvs = calculate_eigvs(kernel, FM_SIZES[conv_ct])
    eigvs = torch.tensor(eigvs)
    
    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
        
    vals, args = torch.sort(eigvs*sum_of_kernel)
    return args[:num_elimination].tolist()

def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))
    
    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor

def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv.bias.data

        return new_conv, residue

def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                out_features=linear.out_features,
                                bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data
    
    return new_linear

def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)
        
    return new_norm

    