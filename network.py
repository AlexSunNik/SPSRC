import torch
import torchvision.models as models

class VGG(torch.nn.Module):
    def __init__(self, vgg='vgg16_bn', data_set='CIFAR10', pretrained=False):
        super(VGG, self).__init__()
        self.features = models.__dict__[vgg](pretrained=pretrained).features
        
        classifier = []
        if 'CIFAR' in data_set:
            num_class = int(data_set.split("CIFAR")[1])
            
#             classifier.append(torch.nn.Linear(512, 512))
#             classifier.append(torch.nn.BatchNorm1d(512))
            classifier.append(torch.nn.Linear(512, num_class))
        else:
            raise RuntimeError("Not expected data flag !!!")

        self.classifier = torch.nn.Sequential(*classifier)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # Print the feature map size
    def test_forward(self, x):
        print(f"Initial Size: {x.shape}")
        print("Begin testing forward pass")
        print("###################################")
        conv_ct = 1
        for i in range(len(self.features)):
            pre_size = x.shape
            x = self.features[i](x)
            pos_size = x.shape
            if isinstance(self.features[i], torch.nn.Conv2d):
                print(f"For conv layer {conv_ct}:")
                print(f"Pre Conv: {pre_size}")
                print(f"Pos Conv: {pos_size}")
                conv_ct += 1
                