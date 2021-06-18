from albumentations import (Compose, Normalize)
from albumentations import (Flip)
from albumentations.pytorch import ToTensorV2

class BaseTrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            # Flip(p=0.5),
            ToTensorV2(p=1.0),
        ], bbox_params={'format': 'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['labels']})

    def __call__(self, image, bboxes, labels):
        return self.transformer(image=image, bboxes=bboxes, labels=labels)


class BaseValAugmentation:
    def __init__(self):
        self.transformer = Compose([
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(p=1.0),
        ], bbox_params={'format': 'pascal_voc', 'min_area': 0, 'min_visibility': 0, 'label_fields': ['labels']})

    def __call__(self, image, bboxes, labels):
        return self.transformer(image=image, bboxes=bboxes, labels=labels)


class BaseTestAugmentation:
    def __init__(self):
        self.transformer = Compose([
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(p=1.0),
        ])

    def __call__(self, image):
        return self.transformer(image=image)


    
def vali_fn(val_data_loader, model, device):
    model.eval()
    vali_loss = AverageMeter()
    vali_mAP = AverageMeter()
    # Custom
    metric = gcv.metrics.VOCMApMetric(iou_thresh=0.5)
    with torch.no_grad():
        for images, targets, image_ids in tqdm(val_data_loader):
            # gpu 계산을 위해 image.to(device)
            images = torch.stack(images).to(device).float()
            current_batch_size = images.shape[0]

            targets2 = {}
            targets2['bbox'] = [target['boxes'].to(device).float() for target in targets] # variable number of instances, so the entire structure can be forced to tensor
            targets2['cls'] = [target['labels'].to(device).float() for target in targets]
            targets2['image_id'] = torch.tensor([target['image_id'] for target in targets]).to(device).float()
            targets2['img_scale'] = torch.tensor([target['img_scale'] for target in targets]).to(device).float()
            targets2['img_size'] = torch.tensor([(512, 512) for target in targets]).to(device).float()

            outputs = model(images, targets2)

            loss = outputs['loss']
            det = outputs['detections']

            # Calc Metric
            for i in range(0, len(det)):
                pred_scores=det[i, :, 4].cpu().unsqueeze_(0).numpy()
                condition=(pred_scores > 0.05)[0]
                gt_boxes=targets2['bbox'][i].cpu().unsqueeze_(0).numpy()[...,[1,0,3,2]] #move to PASCAL VOC from yxyx format
                gt_labels=targets2['cls'][i].cpu().unsqueeze_(0).numpy()

                pred_bboxes=det[i, :, 0:4].cpu().unsqueeze_(0).numpy()[:, condition, :]
                pred_labels=det[i, :, 5].cpu().unsqueeze_(0).numpy()[:, condition]
                pred_scores=pred_scores[:, condition]
                metric.update(
                  pred_bboxes=pred_bboxes,
                  pred_labels=pred_labels,
                  pred_scores=pred_scores,
                  gt_bboxes=gt_boxes,
                  gt_labels=gt_labels)

            vali_mAP.update(metric.get()[1], current_batch_size)
            vali_loss.update(loss.detach().item(), current_batch_size)
    
    # validation loss
    return vali_loss.avg, vali_mAP.avg