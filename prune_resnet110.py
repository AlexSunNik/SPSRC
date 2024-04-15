#!/workspace/alexsun/reconv_compression/mtlc_python/bin/env

from parameter import *
from train import train_network
from evaluate import test_network
from utils import *
from reconv import *
from config import *
from prune import *

# Set a random seed for fair comparison between different metric
fix_random_seed(8)
# fix_random_seed(88)

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
    assert 'resnet' in args.model
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    print(str(args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#     device = torch.device(args.gpu if args.gpu_no >= 0 else "cpu")
    # Load the trained net
    net_name = f"{args.data_set}_model"
    network = load_basenet(args, name=net_name, gpu="cuda:7")
#     network = network.module.cpu()
    network = network.cuda()
    _, _, (top1, top5) = test_network(args, network=network)
    # Prune Config
    skip = [36, 38, 74]
#     if args.prune_cfg == 1:
#         prune_prob = [0.1, 0.1, 0.1]
    if args.prune_cfg == 1:
        prune_prob = [0.5, 0.4, 0.3]
#     elif args.prune_cfg == 3:
#         skip = [16, 18, 20, 34, 38, 54]
#         prune_prob = [0.6, 0.3, 0.1]
#     elif args.prune_cfg == 4:
#         skip = [16, 18, 20, 34, 38, 54]
#         prune_prob = [0.6, 0.5, 0.5]
    elif args.prune_cfg == 2:
        prune_prob = [0.7, 0.6, 0.4]
    else:
        print("Unrecognized Pruning Configuration")
        exit()
    # Get the metric saliency score
    ###################################################
    if args.metric == 'spec':
        print("Prunning By Spectral Norm")
        exec(f"from saliency.{args.model}_{net_name}_eigvs import *")
        prune_eigvs = {}
        for i in range(2, 110, 2):
            prune_eigvs[i] = eval(f"eigvs{i}")
            prune_eigvs[i] = [torch.from_numpy(eigv) for eigv in eval(f"eigvs{i}")]

        network = prune_resnet110(args, network, skip, prune_prob, prune_eigvs)
        base_log = args.save_path+"/logs/"+f"{args.model}_{net_name}_spec_pruned"
        net_name += "spec_pruned"
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
        prune_nucs = {}
        for i in range(2, 110, 2):
            prune_nucs[i] = eval(f"nucs{i}")

        network = prune_resnet110(args, network, skip, prune_prob, prune_nucs)
        base_log = args.save_path+"/logs/"+f"{args.model}_{net_name}_nuc_pruned"
        net_name += "nuc_pruned"
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
        prune_fros = {}
        for i in range(2, 110, 2):
            prune_fros[i] = eval(f"fros{i}")

        network = prune_resnet110(args, network, skip, prune_prob, prune_fros)
        base_log = args.save_path+"/logs/"+f"{args.model}_{net_name}_fro_pruned"
        net_name += "fro_pruned"
        if args.prune_cfg != 1:
            base_log += f"cfg{args.prune_cfg}"
            net_name += f"cfg{args.prune_cfg}"
        
        log_path = get_log_file_name(base_log)
        print(log_path)
        log_file = open(log_path, 'a')
    ###################################################            
    else:
        print("Unrecogrnized Metric Input")
    
    # Retraining
    network.cuda()
    _, _, (top1, top5) = test_network(args, network=network, log_file=log_file)
    print("Pre-finetuning Accuracy:", top1, top5)
    network, acc1, acc5 = retrain(args, network, retrain_epoch=80, save_best=True, net_name=f"{args.model}_{net_name}", log_file=log_file)
    print("Post-finetuning Accuracy:", acc1, acc5)

if __name__ == '__main__':
    main()