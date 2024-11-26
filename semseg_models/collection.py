import torch
from .deepv3 import DeepWV3Plus
from .DualGCNNet import ResNet
from .GALDNet import Bottleneck
from .model_stages import BiSeNet


def DeepLabV3Plus_WideResNet38(num_classes=19):
    model = DeepWV3Plus(num_classes, trunk='WideResnet38')
    return torch.nn.DataParallel(model)


def DualSeg_ResNet50(num_classes=19):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model


def STDC2_Seg75(num_classes=19):
    model = BiSeNet(backbone="STDCNet1446", n_classes=num_classes, use_boundary_2=False, use_boundary_4=False,
                    use_boundary_8=True, use_boundary_16=False, use_conv_last=False)
    return model

def STDC1_Seg50(num_classes=19):
    model = BiSeNet(backbone="STDCNet813", n_classes=num_classes, use_boundary_2=False, use_boundary_4=False,
                    use_boundary_8=True, use_boundary_16=False, use_conv_last=False)
    return model