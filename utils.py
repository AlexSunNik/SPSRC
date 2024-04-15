import torch
import torchvision
import torchvision.transforms as transforms
# from resnet_imagenet import *
from resnet_cifar import *
import random
import numpy as np

class AverageMeter(object):    
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_normalizer(data_set, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)
    elif data_set == 'ImageNet':
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)
    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD = [1/std for std in STD]

    return transforms.Normalize(MEAN, STD)

def get_transformer(data_set, imsize=None, cropsize=None, crop_padding=None, hflip=None):
    transformers = [] 
    if imsize:
        transformers.append(transforms.Resize(imsize))
    if cropsize:
        ## https://github.com/kuangliu/pytorch-cifar
        transformers.append(transforms.RandomCrop(cropsize, padding=crop_padding))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip(hflip))

    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set))
    
    return transforms.Compose(transformers)

def get_data_set(args, train_flag=True):
    if train_flag:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=True, 
                                       transform=get_transformer(args.data_set, args.imsize,
                                           args.cropsize, args.crop_padding, args.hflip), download=True)
    else:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=False, 
                                           transform=get_transformer(args.data_set), download=True)    
    return data_set

def load_basenet(args, name="model", gpu="cuda", full_name=None):
    from network import VGG
    if 'vgg' in args.model:
        network = VGG(args.model, args.data_set)
    else:
        import re
        PAT = re.compile("\d+")
        depth = int(PAT.search(args.model).group(0))
        network = resnet(depth=depth)
#     network = torch.nn.DataParallel(network)
#     device = torch.device(gpu if args.gpu_no >= 0 else "cpu")
    device = torch.device("cpu")
    network = network.to(device)
    if full_name is not None:
        check_point = torch.load(args.save_path+full_name, map_location=device)
    else:
        check_point = torch.load(args.save_path+f"final_{args.model}_{name}.pth", map_location=device)
#     check_point = torch.load("/data/alexsun/save_model/reconv_compression/final_vgg16_bn_model.pth")
    network.load_state_dict(check_point['state_dict'])
    return network

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False