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
from resnet_imagenet import *
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
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#     print(str(args.gpu))
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#     device = torch.device(args.gpu if args.gpu_no >= 0 else "cpu")
    # Load the trained net
    net_name = f"{args.data_set}_model"
    
    model = resnet34(pretrained=True)
    model = model.cuda()
    
    # Prune Config
    skip = [2, 8, 14, 16, 26, 28, 30, 32]
    if args.prune_cfg == 1:
        prune_prob = [0.5, 0.6, 0.4, 0.0]
    elif args.prune_cfg == 2:
        prune_prob = [0.8, 0.8, 0.7, 0.3]
    else:
        print("Unrecognized Pruning Configuration")
        exit()
    # Get the metric saliency score
    ###################################################
    if args.metric == 'spec':
        print("Prunning By Spectral Norm")
        exec(f"from saliency.{args.model}_{net_name}_eigvs import *")
        prune_eigvs = {}
        for i in range(2, 34, 2):
            if i in skip:
                continue
            prune_eigvs[i] = eval(f"eigvs{i}")
            prune_eigvs[i] = [torch.from_numpy(eigv) for eigv in eval(f"eigvs{i}")]
        
        newmodel, cfg = prune_resnet34(args, model, skip, prune_prob, prune_eigvs)
        net_name = f"{args.model}_{args.data_set}_model"
        net_name += "_spec_pruned"
        if args.prune_cfg != 1:
            net_name += f"cfg{args.prune_cfg}"
        
        net_name += ".pth.tar"
        torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save_path, net_name))
    ###################################################
    elif args.metric == 'nuc':
        print("Prunning By Nuclear Norm")
        exec(f"from saliency.{args.model}_{net_name}_nucs import *")
        prune_nucs = {}
        for i in range(2, 34, 2):
            if i in skip:
                continue
            prune_nucs[i] = eval(f"nucs{i}")
        
        newmodel, cfg = prune_resnet34(args, model, skip, prune_prob, prune_nucs)
        
        net_name = f"{args.model}_{args.data_set}_model"
        net_name += "_nuc_pruned"
        if args.prune_cfg != 1:
            net_name += f"cfg{args.prune_cfg}"
        
        net_name += ".pth.tar"
        torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save_path, net_name))
    ###################################################  
    elif args.metric == 'fro':
        print("Prunning By Frobenius Norm")
        exec(f"from saliency.{args.model}_{net_name}_fros import *")
        prune_fros = {}
        for i in range(2, 34, 2):
            if i in skip:
                continue
            prune_fros[i] = eval(f"fros{i}")
        
        newmodel, cfg = prune_resnet34(args, model, skip, prune_prob, prune_fros)
        
        net_name = f"{args.model}_{args.data_set}_model"
        net_name += "_fro_pruned"
        if args.prune_cfg != 1:
            net_name += f"cfg{args.prune_cfg}"
        
        net_name += ".pth.tar"
        torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save_path, net_name))
    ###################################################            
    else:
        print("Unrecogrnized Metric Input")
    

if __name__ == '__main__':
    main()