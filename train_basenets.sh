echo "training vggnet"
python3 train_vgg.py --model vgg16_bn --data-set CIFAR100 --gpu 0
python3 train_resnet.py --model resnet34 --data-set CIFAR100 --gpu 1

