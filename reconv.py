import torch
import numpy as np
from config import *
import torch.nn as nn

def conv_to_mat(kernel_wt, img_s, pad):
    k_s = kernel_wt.shape[0]
    
    A = torch.zeros(img_s**2, (img_s+2*pad)**2).cuda()
    r = torch.zeros((img_s+2*pad)**2).cuda()
    k=0
    del_r = []
    for i in range((img_s+2*pad)**2):
        if(i%(img_s+2*pad)==0 or (i+1)%(img_s + 2*pad)==0):
            del_r.append(i)
        if(i%(img_s+2*pad) == 0 and k<k_s):
            
            for j in range(k_s):
                r[i] = kernel_wt[k][j]
                i+=1
            i-=1
            k+=1
       
    
    for i in range(1, img_s+2*pad-1):
        del_r.append(i)
        del_r.append((img_s+2*pad)**2 -1- i)

    del_r1=[i for i in range((img_s + 2*pad)**2)]
    del_r = list(list(set(del_r1)-set(del_r)) + list(set(del_r)-set(del_r1)))

    
    r_tmp = r.clone()
    k=0
    for i in range(img_s**2):
        A[i] = r_tmp
        r_tmp = torch.roll(r_tmp, 1)
        k+=1
        if(k== img_s):#hard coded for padding 1
            r_tmp = torch.roll(r_tmp, k_s-1)
            k=0


    A =A[:, del_r]
    
    return A

def conv_to_mat_gen(kernel_wt, o_size, i_size, pad=(1,1), stride=(1, 1)):
    """generalized version of above implementation. Works for
    any padding and kernel size.
    kernel_wt - wt of the kernel
    o_size - output spatial size from the convolution operation
    i_size - input spatial size given to the convolution operation
    pad - (padx, pady) where pady - padding in rows pady- padding in columns
    stride - convolution stride in the format (stridex, stridey)
    """
    k_sy, k_sx = kernel_wt.shape
    padx, pady = pad
    stridex, stridey = stride
    A = torch.zeros(o_size**2, (i_size+2*padx)*(i_size+2*pady))
    r = torch.zeros((i_size+2*padx)*(i_size+2*pady))
    k=0

    for i in range((i_size+2*padx)*(i_size+2*pady)):
        
        if(i%(i_size+2*padx) == 0 and k<k_sy):
            
            for j in range(k_sx):
                r[i] = kernel_wt[k][j]
                i+=1
            i-=1
            k+=1
       
    
    retain = np.array([i for i in range((i_size+2*padx)*(i_size+2*pady))])
    retain = np.reshape(retain, (i_size+2*pady, i_size+2*padx))
    retain = retain[pady:pady + i_size, padx:padx + i_size]
    retain = retain.flatten()
    
    r_tmp = r.clone()
    k=0
    for i in range(o_size**2):
        A[i] = r_tmp
        r_tmp = torch.roll(r_tmp, stridex)
        
        if(k + k_sx == i_size + 2*padx):
            r_tmp = torch.roll(r_tmp, k_sx - stridex + (stridey-1)*(i_size+2*padx))
            k=0
            continue
        k+=stridex


    A =A[:, retain]
    return A

def calculate_eigvs(conv_layer, scale, padding=1):
    with torch.no_grad():
        eigvs = []
        eigvs_test = []
        # Generate the distribution of eigen values on the first layer
        # print(test_conv.weight.shape)
        out_channels = conv_layer.shape[0]
        in_channels = conv_layer.shape[1]
        for out_idx in range(out_channels):
#             print(out_idx)
            stack = None
            prod = None
            torch.cuda.empty_cache()
            for c in range(in_channels):
                A = conv_to_mat(conv_layer[out_idx, c], scale, padding)
            #     print(A.shape)
                if stack is None:
                    stack = A
                else:
                    stack = torch.cat((stack, A), dim=1)
#             print(stack.shape)
            # Gram matrix has the same eigenvalues as stack.T mult stack
            prod = torch.matmul(stack, stack.T)
            eigv = torch.lobpcg(prod, k=1)[0].data.detach().cpu().numpy()
            eigvs.append(eigv)
    return eigvs

def calculate_nucs(conv_layer, scale, padding=1):
    with torch.no_grad():
        nucs = []
        # Generate the distribution of eigen values on the first layer
        # print(test_conv.weight.shape)
        out_channels = conv_layer.shape[0]
        in_channels = conv_layer.shape[1]
        for out_idx in range(out_channels):
            print(out_idx)
            stack = None
            torch.cuda.empty_cache()
            for c in range(in_channels):
                A = conv_to_mat(conv_layer[out_idx, c], scale, padding)
            #     print(A.shape)
                if stack is None:
                    stack = A
                else:
                    stack = torch.cat((stack, A), dim=1)
#             print(stack.shape)
            # Gram matrix has the same eigenvalues as stack.T mult stack
#             nuc = torch.linalg.norm(stack.cpu(), ord='nuc')
            nuc = torch.linalg.norm(stack.cpu(), ord='nuc')
            nucs.append(nuc)
    return nucs

def calculate_fros(conv_layer, scale, padding=1):
    with torch.no_grad():
        fros = []
        # Generate the distribution of eigen values on the first layer
        # print(test_conv.weight.shape)
        out_channels = conv_layer.shape[0]
        in_channels = conv_layer.shape[1]
        for out_idx in range(out_channels):
#             print(out_idx)
            stack = None
            torch.cuda.empty_cache()
            for c in range(in_channels):
                A = conv_to_mat(conv_layer[out_idx, c], scale, padding)
            #     print(A.shape)
                if stack is None:
                    stack = A
                else:
                    stack = torch.cat((stack, A), dim=1)
#             print(stack.shape)
            # Gram matrix has the same eigenvalues as stack.T mult stack
            fro = torch.linalg.norm(stack.cpu(), ord='fro')
            fros.append(fro)
    return fros

def calculate_rabs(conv_layer, scale, padding=1):
    with torch.no_grad():
        rabss = []
        # Generate the distribution of eigen values on the first layer
        # print(test_conv.weight.shape)
        out_channels = conv_layer.shape[0]
        in_channels = conv_layer.shape[1]
        for out_idx in range(out_channels):
#             print(out_idx)
            stack = None
            torch.cuda.empty_cache()
            for c in range(in_channels):
                A = conv_to_mat(conv_layer[out_idx, c], scale, padding)
            #     print(A.shape)
                if stack is None:
                    stack = A
                else:
                    stack = torch.cat((stack, A), dim=1)
#             print(stack.shape)
            # Gram matrix has the same eigenvalues as stack.T mult stack
            rabs = torch.sum(torch.abs(stack.flatten()))
            rabss.append(rabs)
    return rabss

def calculate_backward_eigvs(conv_layer, scale, padding=1):
    with torch.no_grad():
        eigvs = []
        eigvs_test = []
        # Generate the distribution of eigen values on the first layer
        # print(test_conv.weight.shape)
        out_channels = conv_layer.shape[0]
        in_channels = conv_layer.shape[1]
        for in_idx in range(in_channels):
            stack = None
            prod = None
            torch.cuda.empty_cache()
            for out_idx in range(out_channels):
                A = conv_to_mat(conv_layer[out_idx, in_idx], scale, padding)
                if stack is None:
                    stack = A
                else:
                    stack = torch.cat((stack, A), dim=0)
#             print(stack.shape)
                
            prod = torch.matmul(stack.T, stack)
            eigv = torch.lobpcg(prod, k=1)[0].data.detach().cpu().numpy()
            eigvs.append(eigv)
            
    return eigvs

# def get_opt_value(conv_layer, scale, padding=1):
#     opt_vals = []
#     with torch.no_grad():
#         # print(test_conv.weight.shape)
#         out_channels = conv_layer.shape[0]
#         in_channels = conv_layer.shape[1]
#         for out_idx in range(out_channels):
# #             print(out_idx)
#             stack = None
#             prod = None
#             torch.cuda.empty_cache()
#             for c in range(in_channels):
#                 A = conv_to_mat(conv_layer[out_idx, c], scale, padding)
#             #     print(A.shape)
#                 if stack is None:
#                     stack = A
#                 else:
#                     stack = torch.cat((stack, A), dim=1)
# #             print(stack.shape)
#             opt_vec = torch.sum(stack, dim=0)
#             lam = torch.norm(opt_vec)
#             opt_vec = opt_vec/lam
# #             out += opt_vec
#             max_val = torch.sum(torch.matmul(stack, opt_vec))
#             # Gram matrix has the same eigenvalues as stack.T mult stack
#             opt_vals.append(max_val)
#     return opt_vals

def test_transform_error(network, test_imgs):
    # Test the error of transformation to matrix multiplication
    test_conv = network.features[0]
    # print(test_conv.weight.shape)
    out_channels = test_conv.weight.shape[0]
    in_channels = test_conv.weight.shape[1]
    out_fm_idx = 0

    stack = None
    for c in range(in_channels):
        A = conv_to_mat(test_conv.weight[out_fm_idx, c], 32, 1)
    #     print(A.shape)
        if stack is None:
            stack = A
        else:
            stack = torch.cat((stack, A), dim=1)
    # print(stack.shape)
    test_img = test_imgs[0]
    test_img = test_img.flatten().unsqueeze(1)
    # print(test_img.shape)
    mat_out = torch.matmul(stack.cpu(), test_img.cpu())
    conv_out = test_conv(test_imgs[0].unsqueeze(0))[0][0]
    conv_out = conv_out.flatten().unsqueeze(1)
    # print(conv_out.shape)
    # Expected a very small error induced by the bias term
    print(torch.sum(mat_out - conv_out).data)

##########################################################################################
## VGG
def compute_save_eigvs(network, file_name = "eigvs.py", conv_idxs=CONV_IDXS):
    f = open(f"./{file_name}", "a")
    with torch.no_grad():
        for i in range(NUM_LAYER):
            print(i,  conv_idxs[i], FM_SIZES[i])
            conv_idx = conv_idxs[i]
            eigvs = calculate_eigvs(network.features[conv_idx].weight.data, FM_SIZES[i])
            f.write(f"eigvs{i}={eigvs}\n")
            eigvs = None
            torch.cuda.empty_cache()
    f.close()
    
def compute_save_nucs(network, file_name = "nucs.py", conv_idxs=CONV_IDXS):
    f = open(f"./{file_name}", "a")
    with torch.no_grad():
        for i in range(NUM_LAYER):
            print(i,  conv_idxs[i], FM_SIZES[i])
            conv_idx = conv_idxs[i]
            nucs = calculate_nucs(network.features[conv_idx].weight.data, FM_SIZES[i])
            f.write(f"nucs{i}={nucs}\n")
            nucs = None
            torch.cuda.empty_cache()
    f.close()
    
def compute_save_fros(network, file_name = "fros.py", conv_idxs=CONV_IDXS):
    f = open(f"./{file_name}", "a")
    with torch.no_grad():
        for i in range(NUM_LAYER):
            print(i,  conv_idxs[i], FM_SIZES[i])
            conv_idx = conv_idxs[i]
            fros = calculate_fros(network.features[conv_idx].weight.data, FM_SIZES[i])
            f.write(f"fros{i}={fros}\n")
            fros = None
            torch.cuda.empty_cache()
    f.close()

def compute_save_rabs(network, file_name = "rabs.py", conv_idxs=CONV_IDXS):
    f = open(f"./{file_name}", "a")
    with torch.no_grad():
        for i in range(NUM_LAYER):
            print(i,  conv_idxs[i], FM_SIZES[i])
            conv_idx = conv_idxs[i]
            rabs = calculate_rabs(network.features[conv_idx].weight.data, FM_SIZES[i])
            f.write(f"rabs{i}={rabs}\n")
            rabs = None
            torch.cuda.empty_cache()
    f.close()

##########################################################################################
## ResNet 56   
def compute_save_eigvs_resnet56(model, file_name = "eigvs.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
#             if layer_id in skip:
#                 layer_id += 1
#                 continue
            if layer_id % 2 == 0:
                if layer_id <= 18:
                    stage = 0
                elif layer_id <= 36:
                    stage = 1
                else:
                    stage = 2
                # Compute Eigvs
                print(layer_id)
                kernel = m.weight.data
                eigvs = calculate_eigvs(kernel, FM_SIZES_RES56[stage])
                f.write(f"eigvs{layer_id}={eigvs}\n")
                eigvs = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()
    
def compute_save_nucs_resnet56(model, file_name = "nucs.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
#             if layer_id in skip:
#                 layer_id += 1
#                 continue
            if layer_id % 2 == 0:
                if layer_id <= 18:
                    stage = 0
                elif layer_id <= 36:
                    stage = 1
                else:
                    stage = 2
                # Compute nucs
                print(layer_id)
                kernel = m.weight.data
                nucs = calculate_nucs(kernel, FM_SIZES_RES56[stage])
                f.write(f"nucs{layer_id}={nucs}\n")
                nucs = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()
    
def compute_save_fros_resnet56(model, file_name = "fros.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
#             if layer_id in skip:
#                 layer_id += 1
#                 continue
            if layer_id % 2 == 0:
                if layer_id <= 18:
                    stage = 0
                elif layer_id <= 36:
                    stage = 1
                else:
                    stage = 2
                # Compute fros
                print(layer_id)
                kernel = m.weight.data
                fros = calculate_fros(kernel, FM_SIZES_RES56[stage])
                f.write(f"fros{layer_id}={fros}\n")
                fros = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()

##########################################################################################
## ResNet 110   
def compute_save_eigvs_resnet110(model, file_name = "eigvs.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
#             if layer_id in skip:
#                 layer_id += 1
#                 continue
            if layer_id % 2 == 0:
                stage = layer_id // 36
                if layer_id <= 36:
                    stage = 0
                elif layer_id <= 72:
                    stage = 1
                elif layer_id <= 108:
                    stage = 2
                # Compute Eigvs
                print(layer_id)
                kernel = m.weight.data
                eigvs = calculate_eigvs(kernel, FM_SIZES_RES56[stage])
                f.write(f"eigvs{layer_id}={eigvs}\n")
                eigvs = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()
    
def compute_save_nucs_resnet110(model, file_name = "nucs.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
#             if layer_id in skip:
#                 layer_id += 1
#                 continue
            if layer_id % 2 == 0:
                stage = layer_id // 36
                if layer_id <= 36:
                    stage = 0
                elif layer_id <= 72:
                    stage = 1
                elif layer_id <= 108:
                    stage = 2
                # Compute nucs
                print(layer_id)
                kernel = m.weight.data
                nucs = calculate_nucs(kernel, FM_SIZES_RES56[stage])
                f.write(f"nucs{layer_id}={nucs}\n")
                nucs = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()
    
def compute_save_fros_resnet110(model, file_name = "fros.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
#             if layer_id in skip:
#                 layer_id += 1
#                 continue
            if layer_id % 2 == 0:
                stage = layer_id // 36
                if layer_id <= 36:
                    stage = 0
                elif layer_id <= 72:
                    stage = 1
                elif layer_id <= 108:
                    stage = 2
                # Compute fros
                print(layer_id)
                kernel = m.weight.data
                fros = calculate_fros(kernel, FM_SIZES_RES56[stage])
                f.write(f"fros{layer_id}={fros}\n")
                fros = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()

##########################################################################################
## ResNet 34   
def compute_save_eigvs_resnet34(model, skip, file_name = "eigvs.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1,1):
                continue
            if layer_id in skip:
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
                # Compute Eigvs
                print(layer_id)
                kernel = m.weight.data
                eigvs = calculate_eigvs(kernel, FM_SIZES_RES34[stage])
                f.write(f"eigvs{layer_id}={eigvs}\n")
                eigvs = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()

def compute_save_nucs_resnet34(model, skip, file_name = "nucs.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1,1):
                continue
            if layer_id in skip:
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
                # Compute Nucs
                print(layer_id)
                kernel = m.weight.data
                nucs = calculate_nucs(kernel, FM_SIZES_RES34[stage])
                f.write(f"nucs{layer_id}={nucs}\n")
                nucs = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()
    
def compute_save_fros_resnet34(model, skip, file_name = "fros.py"):
    f = open(f"./{file_name}", "a")
    layer_id = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1,1):
                continue
            if layer_id in skip:
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
                # Compute Fros
                print(layer_id)
                kernel = m.weight.data
                fros = calculate_fros(kernel, FM_SIZES_RES34[stage])
                f.write(f"fros{layer_id}={fros}\n")
                fros = None
                torch.cuda.empty_cache()
                layer_id += 1
                continue
            layer_id += 1
    f.close()

    
# def compute_save_rabs_resnet(network, file_name = "rabs.py"):
#     f = open(f"./{file_name}", "a")
#     with torch.no_grad():
#         for i in range(NUM_LAYER):
#             print(i,  conv_idxs[i], FM_SIZES[i])
#             conv_idx = conv_idxs[i]
#             rabs = calculate_rabs(network.features[conv_idx].weight.data, FM_SIZES[i])
#             f.write(f"rabs{i}={rabs}\n")
#             rabs = None
#             torch.cuda.empty_cache()
#     f.close()


# def compute_save_resnet_eigvs(network, prune_layers, file_name = "resnet_eigvs.py"):
#     conv_index = 2
#     layers = [network.conv2_x, network.conv3_x, network.conv4_x, network.conv5_x]
#     f = open(f"./{file_name}", "a")
#     with torch.no_grad():
#         for layer_index in range(len(layers)):
#             for block_index in range(len(layers[layer_index])):
#                 if "conv_%d" % conv_index in prune_layers:
#                     # identify channels to remove
# #                     print(layer_index, block_index)
# #                     print(layers[layer_index][block_index].residual_function[0])
#                     kernel = layers[layer_index][block_index].residual_function[0].weight.data
#                     eigvs = calculate_eigvs(kernel, FM_SIZES_RES[layer_index])
#                     print(len(eigvs))
#                     f.write(f"eigvs{conv_index}={eigvs}\n")
#                 conv_index += 2
#     f.close()

# def compute_save_resnet_nucs(network, prune_layers, file_name = "resnet_nucs.py"):
#     conv_index = 2
#     layers = [network.conv2_x, network.conv3_x, network.conv4_x, network.conv5_x]
#     f = open(f"./{file_name}", "a")
#     with torch.no_grad():
#         for layer_index in range(len(layers)):
#             for block_index in range(len(layers[layer_index])):
#                 if "conv_%d" % conv_index in prune_layers:
#                     # identify channels to remove
# #                     print(layer_index, block_index)
# #                     print(layers[layer_index][block_index].residual_function[0])
#                     kernel = layers[layer_index][block_index].residual_function[0].weight.data
#                     nucs = calculate_nucs(kernel, FM_SIZES_RES[layer_index])
#                     print(len(nucs))
#                     f.write(f"nucs{conv_index}={nucs}\n")
#                 conv_index += 2
#     f.close()

# def compute_save_resnet_fros(network, prune_layers, file_name = "resnet_fros.py"):
#     conv_index = 2
#     layers = [network.conv2_x, network.conv3_x, network.conv4_x, network.conv5_x]
#     f = open(f"./{file_name}", "a")
#     with torch.no_grad():
#         for layer_index in range(len(layers)):
#             for block_index in range(len(layers[layer_index])):
#                 if "conv_%d" % conv_index in prune_layers:
#                     # identify channels to remove
# #                     print(layer_index, block_index)
# #                     print(layers[layer_index][block_index].residual_function[0])
#                     kernel = layers[layer_index][block_index].residual_function[0].weight.data
#                     fros = calculate_fros(kernel, FM_SIZES_RES[layer_index])
#                     print(len(fros))
#                     f.write(f"fros{conv_index}={fros}\n")
#                 conv_index += 2
#     f.close()
    
# def compute_save_resnet_backward_eigvs(network, prune_layers, file_name = "resnet_backward_eigvs.py"):
#     conv_index = 2
#     layers = [network.conv2_x, network.conv3_x, network.conv4_x, network.conv5_x]
#     f = open(f"./{file_name}", "a")
#     with torch.no_grad():
#         for layer_index in range(len(layers)):
#             for block_index in range(len(layers[layer_index])):
#                 if "conv_%d" % conv_index in prune_layers:
#                     # identify channels to remove
# #                     print(layer_index, block_index)
# #                     print(layers[layer_index][block_index].residual_function[0])
# #                     kernel = layers[layer_index][block_index].residual_function[0].weight.data

#                     # Pick the second conv layer
#                     kernel = layers[layer_index][block_index].residual_function[3].weight.data
#                     eigvs = calculate_backward_eigvs(kernel, FM_SIZES_RES[layer_index])
#                     print(len(eigvs))
#                     f.write(f"backward_eigvs{conv_index}={eigvs}\n")
#                 conv_index += 2
#     f.close()
    
# def compute_save_optvs(network, file_name = "optvs.py", conv_idxs=CONV_IDXS_NOBN):
#     f = open(f"./{file_name}", "a")
#     with torch.no_grad():
#         for i in range(NUM_LAYER):
#             print(i,  conv_idxs[i], FM_SIZES[i])
#             conv_idx = conv_idxs[i]
#             optvs = get_opt_value(network.features[conv_idx].weight.data, FM_SIZES[i])
#             f.write(f"optvs{i}={optvs}\n")
#             optvs = None
#             torch.cuda.empty_cache()
#     f.close()