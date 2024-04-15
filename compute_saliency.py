import os
import argparse

import torch
from config import *
from utils import *
from reconv import *

def build_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_no', type=int,
                    help='cpu: -1, gpu: 0 ~ n ', default=NGPU)

#     parser.add_argument('--vgg', type=str,
#                     help='version of vgg network', default=VGG_NAME)
    
    parser.add_argument('--model', type=str,
                    help='version of network', default=VGG_NAME)
    

    parser.add_argument('--save-path',type=str,
                    help='model save path', default=SAVE_PATH)

    parser.add_argument('--prune-layers', nargs='+',
                    help='layer index for pruning', default=None)

    parser.add_argument('--gpu', type=int,
                    help='which gpu to use', default=0)
    
    parser.add_argument('--data-set', type=str,
                    help='Data set for training network', default=DATASET)
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)
    net_name = f"{args.data_set}_model"
    if args.model == 'resnet34':
        from resnet_imagenet import resnet34
        print("Use the Pretrained Model")
        network = resnet34(pretrained=True)
        network = network.cuda()
        network.eval()
    else:
        network = load_basenet(args, name=net_name)
        network = network.cpu()
        network.eval()
    if 'vgg' in args.model:
        # VGG
        # Spectral Norm
        print("Calculating Spectral Norm")
        compute_save_eigvs(network, file_name=f"{args.model}_{net_name}_eigvs.py", conv_idxs=CONV_IDXS)
        # Nuclear Norm
        print("Calculating Nuclear Norm")
        compute_save_nucs(network, file_name=f"{args.model}_{net_name}_nucs.py", conv_idxs=CONV_IDXS)
        # Frobenius Norm
        print("Calculating Frobenius Norm")
        compute_save_fros(network, file_name=f"{args.model}_{net_name}_fros.py", conv_idxs=CONV_IDXS)

    elif args.model == 'resnet56':
        # ResNet 56
        print("Computing Saliency for ResNet56")
        # Spectral Norm
        print("Calculating Spectral Norm")
        compute_save_eigvs_resnet56(network, file_name=f"{args.model}_{net_name}_eigvs.py")
        # Nuclear Norm
        print("Calculating Nuclear Norm")
        compute_save_nucs_resnet56(network, file_name=f"{args.model}_{net_name}_nucs.py")
        # Frobenius Norm
        print("Calculating Frobenius Norm")
        compute_save_fros_resnet56(network, file_name=f"{args.model}_{net_name}_fros.py")
    elif args.model == 'resnet110':
        # ResNet 110
        print("Computing Saliency for ResNet110")
        # Spectral Norm
        print("Calculating Spectral Norm")
        compute_save_eigvs_resnet110(network, file_name=f"{args.model}_{net_name}_eigvs.py")
        # Nuclear Norm
        print("Calculating Nuclear Norm")
        compute_save_nucs_resnet110(network, file_name=f"{args.model}_{net_name}_nucs.py")
        # Frobenius Norm
        print("Calculating Frobenius Norm")
        compute_save_fros_resnet110(network, file_name=f"{args.model}_{net_name}_fros.py")
    elif args.model == 'resnet34':
        # ResNet 110
        print("Computing Saliency for ResNet34")
        skip = [2, 8, 14, 16, 26, 28, 30, 32]
        # Spectral Norm
        print("Calculating Spectral Norm")
        compute_save_eigvs_resnet34(network, file_name=f"{args.model}_{net_name}_eigvs.py", skip=skip)
        # Nuclear Norm
        print("Calculating Nuclear Norm")
        compute_save_nucs_resnet34(network, file_name=f"{args.model}_{net_name}_nucs.py", skip=skip)
        # Frobenius Norm
        print("Calculating Frobenius Norm")
        compute_save_fros_resnet34(network, file_name=f"{args.model}_{net_name}_fros.py", skip=skip)
        
if __name__ == '__main__':
    main()