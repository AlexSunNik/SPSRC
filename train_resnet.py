from parameter import *
from train import train_network
from evaluate import test_network
from resnet_cifar import *

parser = build_parser()
args = parser.parse_args()

assert args.train_flag == 1
assert 'resnet' in args.model

print("Use GPU:", args.gpu)
print("Dataset:", args.data_set)
print("Model Version:", args.model)
print(args.lr_milestone)
print(args.lr_gamma)
print(args.epoch)

import re
PAT = re.compile("\d+")
depth = int(PAT.search(args.model).group(0))

if args.data_set == "CIFAR100":
    net = resnet(depth=depth, dataset='cifar100')
elif args.data_set == "CIFAR10":
    net = resnet(depth=depth)
else:
    print("Unrecognized Dataset")
net_name=f"{args.data_set}_model"
log_file = open(args.save_path+"/logs/"+f"{args.model}_{net_name}.txt", 'a')
network = train_network(args, network=net, net_name=net_name, gpu=args.gpu, log_file=log_file)

# Verify from the stored model
_, _, (acc1, acc5) = test_network(args, network=network)
print("Test Accuracy:",acc1, acc5)

log_file.close()