#!/workspace/alexsun/reconv_compression/mtlc_python/bin/env

from parameter import *
from train import train_network
from evaluate import test_network
from utils import *
from reconv import *
from config import *
from prune import *

# Set a random seed for fair comparison between different metric
# fix_random_seed(8)
# fix_random_seed(88)
fix_random_seed(888)
from train import retrain

def get_log_file_name(base_file_name):
    file_name = base_file_name
    count = 1
    while os.path.isfile(file_name + ".txt"):
        count += 1
        file_name = base_file_name
        file_name += f"{count}th_run"
    return file_name + ".txt"

def main():
    parser = build_parser()
    args = parser.parse_args()
    assert args.metric is not None
    assert 'vgg' in args.model
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    print(str(args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    torch.cuda.set_device(7)
#     device = torch.device(args.gpu if args.gpu_no >= 0 else "cpu")
    # Load the trained net
    net_name = f"{args.data_set}_model"
    rev_flag = args.rev
    print("Reverse:", rev_flag)
    network = load_basenet(args, name=net_name, gpu="cuda:7")
    print("Base Network")
    print_model_param_flops(network.cpu(), input_res=32)
    print_model_param_nums(network.cpu())
#     network = network.module.cpu()
    network = network.cuda()
    _, _, (top1, top5) = test_network(args, network=network)
    # Prune Config
    prune_layers = ['conv1', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13']
    if args.prune_cfg == 1:
        prune_channels = [32, 256, 256, 256, 256, 256, 256]
    elif args.prune_cfg == 2:
        prune_channels = [32, 384, 384, 384, 384, 384, 384]
    elif args.prune_cfg == 3:
        prune_channels = [32, 448, 448, 448, 448, 448, 448]
    else:
        print("Unrecognized Pruning Configuration")
        exit()
    # Get the metric saliency score
    ###################################################
    if args.metric == 'spec':
        print("Prunning By Spectral Norm")
        exec(f"from saliency.{args.model}_{net_name}_eigvs import *")
        all_eigvs = []
        for i in range(NUM_LAYER):
            all_eigvs.append(eval(f"eigvs{i}"))
            all_eigvs[i] = [torch.from_numpy(eigv) for eigv in eval(f"eigvs{i}")]
        prune_eigvs = [all_eigvs[0], all_eigvs[7], all_eigvs[8], all_eigvs[9], all_eigvs[10], all_eigvs[11], all_eigvs[12]]
        network = prune(network, prune_layers, prune_channels, prune_eigvs=prune_eigvs, magnitude=False, rev=rev_flag)
        base_log = args.save_path+"/logs/"+f"{args.model}_{net_name}_spec_pruned"
        net_name += "spec_pruned"
        if rev_flag:
            net_name += "_rev"
            base_log += "_rev"
        if args.prune_cfg != 1:
            base_log += f"cfg{args.prune_cfg}"
            net_name += f"cfg{args.prune_cfg}"
            
        log_path = get_log_file_name(base_log)
        print(log_path)
        log_file = open(log_path, 'a')
    ###################################################
    elif args.metric == 'nuc':
        print("Prunning By Nuclear Norm")
        exec(f"from saliency.{args.model}_{net_name}_nucs import *")
        all_nucs = []
        for i in range(NUM_LAYER):
            all_nucs.append(eval(f"nucs{i}"))
        prune_nucs = [all_nucs[0], all_nucs[7], all_nucs[8], all_nucs[9], all_nucs[10], all_nucs[11], all_nucs[12]]
        network = prune(network, prune_layers, prune_channels, prune_eigvs=prune_nucs, magnitude=False, rev=rev_flag)
        base_log = args.save_path+"/logs/"+f"{args.model}_{net_name}_nuc_pruned"
        net_name += "nuc_pruned"
        if rev_flag:
            net_name += "_rev"
            base_log += "_rev"
        if args.prune_cfg != 1:
            base_log += f"cfg{args.prune_cfg}"
            net_name += f"cfg{args.prune_cfg}"
            
        log_path = get_log_file_name(base_log)
        print(log_path)
        log_file = open(log_path, 'a')
    ###################################################  
    elif args.metric == 'fro':
        print("Prunning By Frobenius Norm")
        exec(f"from saliency.{args.model}_{net_name}_fros import *")
        all_fros = []
        for i in range(NUM_LAYER):
            all_fros.append(eval(f"fros{i}"))
        prune_fros = [all_fros[0], all_fros[7], all_fros[8], all_fros[9], all_fros[10], all_fros[11], all_fros[12]]
        network = prune(network, prune_layers, prune_channels, prune_eigvs=prune_fros, magnitude=False, rev=rev_flag)
        base_log = args.save_path+"/logs/"+f"{args.model}_{net_name}_fro_pruned"
        net_name += "fro_pruned"
        if rev_flag:
            net_name += "_rev"
            base_log += "_rev"
        if args.prune_cfg != 1:
            base_log += f"cfg{args.prune_cfg}"
            net_name += f"cfg{args.prune_cfg}"
        
        log_path = get_log_file_name(base_log)
        print(log_path)
        log_file = open(log_path, 'a')
    ###################################################            
    else:
        print("Unrecogrnized Metric Input")
    
    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            print(i)
            kernel = network.features[i].weight
            print(kernel.shape)
    print("Pruned Network")
    print_model_param_flops(network.cpu(), input_res=32)
    print_model_param_nums(network.cpu())
    # Retraining
    network.cuda()
    _, _, (top1, top5) = test_network(args, network=network, log_file=log_file)
    print("Pre-finetuning Accuracy:", top1, top5)
    network, acc1, acc5 = retrain(args, network, retrain_epoch=80, save_best=True, net_name=f"{args.model}_{net_name}", log_file=log_file)
    print("Post-finetuning Accuracy:", acc1, acc5)

if __name__ == '__main__':
    main()