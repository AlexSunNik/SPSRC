from parameter import *
from train import train_network
from evaluate import test_network

parser = build_parser()
args = parser.parse_args()

assert args.train_flag == 1
assert 'vgg' in args.model

print(args.gpu)
print(args.data_set)
print(args.model)
net_name=f"{args.data_set}_model"
log_file = open(args.save_path+"/logs/"+f"{args.model}_{net_name}.txt", 'a')
network = None
network = train_network(args, network=network, net_name=net_name, gpu=args.gpu, log_file=log_file)

# Verify from the stored model
from network import VGG
network = VGG(args.model, args.data_set)
device = torch.device(args.gpu if args.gpu_no >= 0 else "cpu")
network = network.to(device)
check_point = torch.load(args.save_path+f"final_{args.model}_{net_name}.pth")
network.load_state_dict(check_point['state_dict'])

_, _, (acc1, acc5) = test_network(args, network=network, log_file=log_file)
print("Test Accuracy:",acc1, acc5)

log_file.close()