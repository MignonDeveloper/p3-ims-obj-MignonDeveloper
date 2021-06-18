import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO
import numpy as np
import cv2
import os


class RecycleDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transform: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir, mode='train', transform=None):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
            anns = self.coco.loadAnns(ann_ids)

            boxes = np.array([x['bbox'] for x in anns])

            # boxes (x_min, y_min, x_max, y_max)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            labels = np.array([x['category_id'] for x in anns])
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            areas = np.array([x['area'] for x in anns])
            areas = torch.as_tensor(areas, dtype=torch.float32)
                                    
            is_crowds = np.array([x['iscrowd'] for x in anns])
            is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

            target = {
                'boxes': boxes,
                'labels': labels
            }

            # Multi Scale
            # target['img_size'] = torch.tensor([(512, 512)])
            # target['img_scale'] = torch.tensor([1.])

            # transform
            if self.transform:
                for i in range(10):
                    sample = {
                        'image': image,
                        'bboxes': target['boxes'],
                        'labels': labels
                    }
                    transformed = self.transform(**sample)
                    if len(transformed['bboxes']) > 0:
                        image = transformed['image']
                        target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                        target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                        break

            return image, target, image_info['file_name']

        if self.mode == 'test':
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            return image, image_info['file_name']
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())