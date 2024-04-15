TRAIN_FLAG = True
DATASET = 'CIFAR100'
# DATASET = "ImageNet"
VGG_NAME = 'vgg16_bn'
# VGG_NAME = 'vgg16'
RES_NAME = 'resnet34'
DATASET_PATH = '/data/alexsun/'
SAVE_PATH = '/data/alexsun/save_model/reconv_compression/'
LOAD_PATH = SAVE_PATH+"check_point.pth"
# NGPU = 8
NGPU = 1
BATCH_SIZE = 128 * NGPU
NUM_WORKERS = 8
RESUME_FLAG = False
FM_SIZES = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
FM_SIZES_RES = [32, 16, 8, 4]
FM_SIZES_RES56 = [32, 16, 8]
FM_SIZES_RES34 = [56, 28, 14, 7]

CONV_IDXS = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
CONV_IDXS_NOBN = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

NUM_CHANNELS = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
PRUNE_PCTS= [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

RETRAIN_EPOCH = 20
NUM_LAYER = 13
LR = 0.1
# LR = 0.01