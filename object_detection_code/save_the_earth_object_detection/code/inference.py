import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import odach as oda

from importlib import import_module
import os, argparse
from tqdm import tqdm
import gc

from recycle_dataset import RecycleDataset
from utils import seed_everything, get_test_config

class CFG:
    PROJECT_PATH = "/opt/ml/save_the_earth"
    BASE_DATA_PATH = '/opt/ml/input/data'

    # environment_parameters
    coco_test_json = 'test.json'

    # test parameters
    batch_size = 16
    seed = 42
    num_workers = 4
    score_threshold = 0.05

    model = "EfficientDet6"
    test_augmentation = "BaseTestAugmentation"
    model_name = "/EfficientDet6/EfficientDet6_1.bin"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    docs_path = 'docs'
    models_path = 'models'


def get_data_utils():
    def collate_fn(batch):
        return tuple(zip(*batch))

    # get albumentation transformer for test dataset from augmentation.py
    test_transform_module = getattr(import_module("augmentation"), CFG.test_augmentation)
    test_transform = test_transform_module()

    # get train & valid dataset from recycle_dataset.py
    test_dataset = RecycleDataset(data_dir=CFG.BASE_DATA_PATH,
                                  annotation=CFG.coco_test_json,
                                  mode='test',
                                  transform=test_transform)

    # define data loader based on test dataset
    test_loader = DataLoader(test_dataset,
                           batch_size=CFG.batch_size,
                           shuffle=False,
                           num_workers=CFG.num_workers,
                           pin_memory=True,
                           collate_fn=collate_fn)

    return test_dataset, test_loader


# get saved model for inference
def get_model():
    # 미리 저장된 model의 구조를 가지는 모델을 recylce_model.py에서 가져옵니다.
    model_module = getattr(import_module("recycle_model"), CFG.model)
    model = model_module(num_classes=11, checkpoint_path=os.path.join(CFG.models_path, CFG.model_name))

    # 미리 저장된 모델의 정보를 그대로 load
    model.cuda()

    return model


# make predictions
def inference(model, test_loader):
    print('Start Inference.')
    file_name_list = []
    results = []

    model.eval() # to eval modes
    with torch.no_grad():
        for images, image_name in tqdm(test_loader):
            images = torch.stack(images).to(CFG.device).float()
            batch_size = images.shape[0]

            img_info = {
                'img_scale': torch.tensor([1]*batch_size, dtype=torch.float).to(CFG.device),
                'img_size': torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(CFG.device) 
            }
            det = model(images, img_info) # forward pass

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                labels = det[i].detach().cpu().numpy()[:,5]
                indexes = np.where(scores > CFG.score_threshold)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                results.append({
                    'boxes': boxes[indexes],
                    'scores': scores[indexes],
                    'labels': labels[indexes]
                })

            # batch 별 이미지 정보 저장
            file_name_list.extend(image_name)

    print("End prediction.")
    
    # PredictionString 대입
    prediction_strings = []
    for i, output in enumerate(results):
        prediction_string = ''
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            prediction_string += f"{label} {score} {box[0]} {box[1]} {box[2]} {box[3]} "
        prediction_strings.append(prediction_string)

    # submission.csv로 저장
    submission = pd.DataFrame()
    submission['image_id'] = file_name_list
    submission['PredictionString'] = prediction_strings
    submission.to_csv(os.path.join(CFG.docs_path, 'results', f'{CFG.model_name}.csv'), index=False)
    print(submission.head())


def main():
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 개수: {torch.cuda.device_count()}")

    parser = argparse.ArgumentParser(description="Recycle Object Detection")
    parser.add_argument("--config", type=str, default="base_config.json", help=f'test config file (defalut: base_config.json)')
    args = parser.parse_args()

    # parsing config class from custom config.json file
    get_test_config(CFG, os.path.join('..', 'configs', 'test', args.config))
    
    # set every random seed
    seed_everything(CFG.seed)

    # get pytorch data utils (dataset, dataloader)
    test_dataset, test_loader = get_data_utils()

    # load trained model
    model = get_model()

    # inference
    inference(model, test_loader)


if __name__ == "__main__":
    main()