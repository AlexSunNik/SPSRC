import time

import torch
import torchvision

from network import VGG
from loss import Loss_Calculator
from evaluate import accuracy, test_network
from utils import AverageMeter, get_data_set
from optimizer import get_optimizer
from config import RETRAIN_EPOCH

def train_network(args, network=None, data_set=None, net_name="model", gpu="cuda", log_file=None):
    device = torch.device(gpu if args.gpu_no >= 0 else "cpu")
    if network is None:
        network = VGG(args.model, args.data_set)
    if args.gpu_no > 1:
        network = torch.nn.DataParallel(network)
    network = network.to(device)
    
    if data_set is None:
        data_set = get_data_set(args, train_flag=True)
    
    loss_calculator = Loss_Calculator()
    
    optimizer, scheduler = get_optimizer(network, args)
    
    torch.save({'epoch': 0, 
                   'state_dict': network.state_dict(),
                   'loss_seq': loss_calculator.loss_seq},
                   args.save_path+f"init_{args.model}_{net_name}.pth")
    
    if args.resume_flag:
        check_point = torch.load(args.load_path)
        network.load_state_dict(check_point['state_dict'])
        loss_calculator.loss_seq = check_point['loss_seq']
        args.start_epoch = check_point['epoch'] # update start epoch
                
    print("-*-"*10 + "\n\tTrain network\n" + "-*-"*10)
    if log_file is not None:
        log_file.write("-*-"*10 + "\n\tTrain network\n" + "-*-"*10)
        log_file.write("\n")
                       
    for epoch in range(args.start_epoch, args.epoch):
        # make shuffled data loader
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

        # train one epoch
        train_step(network, data_loader, loss_calculator, optimizer, device, epoch, args.print_freq, log_file=log_file)

        # adjust learning rate
        if scheduler is not None:
            scheduler.step()
        
        torch.save({'epoch': epoch+1, 
                   'state_dict': network.state_dict(),
                   'loss_seq': loss_calculator.loss_seq},
                   args.save_path+"check_point.pth")
        
        if (epoch+1) % 100 == 0:
            torch.save({'epoch': epoch+1, 
                   'state_dict': network.state_dict(),
                   'loss_seq': loss_calculator.loss_seq},
                   args.save_path+f"{epoch+1}th_{args.model}_{net_name}.pth")
            
    torch.save({'epoch': epoch+1, 
                   'state_dict': network.state_dict(),
                   'loss_seq': loss_calculator.loss_seq},
                   args.save_path+f"final_{args.model}_{net_name}.pth")
    return network

def train_step(network, data_loader, loss_calculator, optimizer, device, epoch, print_freq=100, log_file=None):
    network.train()
    # set benchmark flag to faster runtime
    torch.backends.cudnn.benchmark = True
        
    data_time = AverageMeter()
    loss_time = AverageMeter()    
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    tic = time.time()
    for iteration, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - tic)
        
        inputs, targets = inputs.to(device), targets.to(device)
        torch.cuda.empty_cache()
        tic = time.time()
        outputs = network(inputs)
        forward_time.update(time.time() - tic)
        
        tic = time.time()
        loss = loss_calculator.calc_loss(outputs, targets)
        loss_time.update(time.time() - tic)
        
        tic = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time.update(time.time() - tic)
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1,5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
                        
        if iteration % print_freq == 0:
            logs_ = '%s: '%time.ctime()
            logs_ += 'Epoch [%d], '%epoch
            logs_ += 'Iteration [%d/%d/], '%(iteration, len(data_loader))
            logs_ += 'Data(s): %2.3f, Loss(s): %2.3f, '%(data_time.avg, loss_time.avg)
            logs_ += 'Forward(s): %2.3f, Backward(s): %2.3f, '%(forward_time.avg, backward_time.avg)
            logs_ += 'Top1: %2.3f, Top5: %2.4f, '%(top1.avg, top5.avg)
            logs_ += 'Loss: %2.3f'%loss_calculator.get_loss_log()
            print(logs_)            
            if log_file is not None:
                log_file.write(logs_)
                log_file.write("\n")
        tic = time.time()
    return None

def retrain(args, network, data_set=None, retrain_epoch=RETRAIN_EPOCH, save_best=False, gpu="cuda", net_name="model", log_file=None):
    device = torch.device(gpu if args.gpu_no >= 0 else "cpu")
#     if args.gpu_no > 1:
#         network = torch.nn.DataParallel(network)
    network = network.to(device)
    
    if data_set is None:
        data_set = get_data_set(args, train_flag=True)
    
    loss_calculator = Loss_Calculator()
    
    # Adjust the lr
    args.lr /= 100
    optimizer, scheduler = get_optimizer(network, args)

    best_acc1 = 0
    best_acc5 = 0
    
    print("-*-"*10 + "\n\tTrain network\n" + "-*-"*10)
    if log_file is not None:
        log_file.write("-*-"*10 + "\n\tTrain network\n" + "-*-"*10)
        log_file.write("\n")
    for epoch in range(retrain_epoch):
        network.train()
        # make shuffled data loader
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

        # train one epoch
        train_step(network, data_loader, loss_calculator, optimizer, device, epoch, args.print_freq, log_file)

        # adjust learning rate
        if scheduler is not None:
            scheduler.step()  
        if epoch == 20 or epoch == 40:
            print("Adjust Learning Rate")
            if log_file is not None:
                log_file.write("Adjust Learning Rate")
                log_file.write("\n")
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10
        if save_best:
            network.eval()
            with torch.no_grad():
                _,_, (acc1, acc5) = test_network(args, network=network, log_file=log_file, gpu=gpu)
                if acc1 > best_acc1:
                    best_acc1 = acc1
                    best_acc5 = acc5
#             torch.save({'epoch': epoch+1, 
#                    'state_dict': network.state_dict(),
#                    'loss_seq': loss_calculator.loss_seq},
#                    args.save_path+f"best_{args.model}_{net_name}.pth")
            
    return network, best_acc1, best_acc5