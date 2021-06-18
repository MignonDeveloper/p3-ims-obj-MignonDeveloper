import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from effdet import EfficientDet, get_efficientdet_config, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet


class torchvision_Faster_RCNN_resnet50_fpn(nn.Module):
    '''
        Backbone: resnet50
        num_class: segmentation하고 싶은 객체의 종류        
        forward output
            - output  : [batch_size, num_classes, height, width]
    '''
    def __init__(self, num_classes=11):
        super(torchvision_Faster_RCNN_resnet50_fpn, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, target=None):
        if target == None:
            output = self.model(x)
        else:
            output = self.model(x, target)
        return output


class EfficientDet6(nn.Module):
    def __init__(self, num_classes=11, checkpoint=None):
        super(EfficientDet6, self).__init__()
        config = get_efficientdet_config('tf_efficientdet_d6')

        config.image_size = [512, 512]
        config.norm_kwargs=dict(eps=.001, momentum=.01)
        config.soft_nms = True
        config.label_smoothing = 0.1
        config.mean = [0.46009655, 0.43957878, 0.41827092]
        config.std = [0.2108204, 0.20766491, 0.21656131]

        net = EfficientDet(config, pretrained_backbone=False)
        if checkpoint == None:
            checkpoint = torch.load('./effdet_model/tf_efficientdet_d6_52-4eda3773.pth')

        net.load_state_dict(checkpoint)

        net.reset_head(num_classes=num_classes)
        net.class_net = HeadNet(config, num_outputs=config.num_classes)

        self.model = DetBenchTrain(net, config)

    
    def forward(self, images, targets):
        return self.model(images, targets)


class EfficientDet5AP(nn.Module):
    def __init__(self, num_classes=11, checkpoint_path=None):
        super(EfficientDet5AP, self).__init__()
        config = get_efficientdet_config('tf_efficientdet_d5_ap')

        config.image_size = [512,512]
        config.norm_kwargs=dict(eps=.001, momentum=.01)
        config.soft_nms = True
        config.label_smoothing = 0.1
        # config.legacy_focal = True

        net = EfficientDet(config, pretrained_backbone=False)
        
        if checkpoint_path == None:
            checkpoint = torch.load('./effdet_model/tf_efficientdet_d5_ap-3673ae5d.pth')
            net.load_state_dict(checkpoint)
            net.reset_head(num_classes=num_classes)
            net.class_net = HeadNet(config, num_outputs=config.num_classes)
            self.model = DetBenchTrain(net, config)

        else:
            checkpoint = torch.load(checkpoint_path)
            checkpoint2 = {'.'.join(k.split('.')[2:]): v for k,v in checkpoint.items()}
            del checkpoint2['boxes']

            net.reset_head(num_classes=num_classes)
            net.class_net = HeadNet(config, num_outputs=config.num_classes)
            
            net.load_state_dict(checkpoint2)
            self.model = DetBenchPredict(net)
            
    
    def forward(self, images, targets):
        return self.model(images, targets)


if __name__ == "__main__":
    # test model forward with simple example
    model = EfficientDet5AP(11)
    # print(model)

