import warnings
warnings.filterwarnings('ignore')

# import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import argparse
import os, time
from tqdm import tqdm
from importlib import import_module

from recycle_dataset import RecycleDataset
from optimizer import create_optimizer
from scheduler import create_scheduler
from utils import EarlyStopping, Averager, seed_everything, get_train_config


class CFG:
    PROJECT_PATH = "/opt/ml/save_the_earth"
    BASE_DATA_PATH = '/opt/ml/input/data'

    # environment_parameters
    coco_train_json = 'train.json'
    coco_val_json = 'val.json'

    # hyper_parameters
    learning_rate = 1e-4
    train_batch_size = 32
    valid_batch_size = 32
    nepochs = 30
    patience = 5
    seed = 42
    num_workers = 4

    # model_parameters
    model = "EfficientDet6"
    optimizer = "AdamW"
    scheduler = "StepLR"
    train_augmentation = "BaseTrainAugmentation"
    val_augmentation = "BaseValAugmentation"
    kfold = 0
    print_freq = 1
    model_save_name = "EfficientDet6"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    docs_path = 'docs'
    models_path = 'models'


def make_folder_structure():
    """
        make default folder structure for training
    """
    os.mkdir(CFG.docs_path, exist_ok=True)
    os.mkdir(CFG.models_path, exist_ok=True)
    os.mkdir(os.path.join(CFG.models_path, CFG.model_save_name), exist_ok=True)
    os.mkdir(os.path.join(CFG.docs_path, 'results'), exist_ok=True)


def get_data_utils():
    """
        define train/validation pytorch dataset & loader

        Returns:
            train_dataset: pytorch dataset for train data
            val_dataset: pytorch dataset for validation data
            train_loader: pytorch data loader for train data
            val_loader: pytorch data loader for validation data
    """    
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # get albumentation transformers for each train & valid dataset from augmentation.py
    train_transform_module = getattr(import_module("augmentation"), CFG.train_augmentation)
    val_transform_module = getattr(import_module("augmentation"), CFG.val_augmentation)
    train_transform = train_transform_module()
    val_transform = val_transform_module()

    # get train & valid dataset from dataset.py
    train_dataset = RecycleDataset(data_dir=CFG.BASE_DATA_PATH,
                                   annotation=CFG.coco_train_json,
                                   mode='train',
                                   transform=train_transform)
    val_dataset = RecycleDataset(data_dir=CFG.BASE_DATA_PATH,
                                 annotation=CFG.coco_val_json,
                                 mode='val',
                                 transform=val_transform)

    # define data loader based on each dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.train_batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn)

    return train_dataset, val_dataset, train_loader, val_loader


def get_model():
    '''
        get defined model from recycle_model.py
        
        Returns:
            model: pytorch model that would be trained
            optimizer: pytorch optimizer for gradient descent
            scheduler: pytorch lr scheduler
    '''
    model_module = getattr(import_module("recycle_model"), CFG.model)
    model = model_module(num_classes=11)

    # move model to cuda memory
    model.cuda()

    # watch model in wandb
    # wandb.watch(model)
    
    # check the number of model parameters
    print('parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # if using multi-gpu, train model in parallel
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # setting weight_decay different
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]  

    # get optimizer from optimizer.py
    optimizer = create_optimizer(
        CFG.optimizer,
        params = optimizer_grouped_parameters,
        lr = CFG.learning_rate,
        **CFG.optimizer_params)

    # get scheduler from scheduler.py
    scheduler = create_scheduler(
        CFG.scheduler,
        optimizer = optimizer,
        **CFG.scheduler_params)

    return model, optimizer, scheduler


def validation(model, val_loader):
    '''
        evaluation function for validation data
    '''

    print("Start validation.\n")
    val_loss_hist = Averager()

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = torch.stack(images).to(CFG.device).float()
            bboxes = [target['boxes'].to(CFG.device).float() for target in targets]
            labels = [target['labels'].to(CFG.device).float() for target in targets]
            batch_size = images.shape[0]

            target_res = dict()
            target_res['bbox'] = bboxes
            target_res['cls'] = labels 
            # target_res["img_scale"] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(CFG.device)
            # target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(CFG.device)
            
            # forward pass & calculate loss
            output = model(images, target_res)
            loss_value = output['loss'].detach().item()
            val_loss_hist.update(loss_value, batch_size)


            del images, targets, bboxes

    return val_loss_hist.value


def train(model, optimizer, scheduler, train_loader, val_loader):
    '''
        train model
    '''

    print("Start training.\n")
    early_stopping = EarlyStopping(patience=CFG.patience, verbose=True, trace_func=print)
    train_loss_hist = Averager()

    for epoch in tqdm(range(1, CFG.nepochs + 1)):
        now = time.localtime()
        print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

        model.train()
        train_loss_hist.reset()

        for step, (images, targets, _) in enumerate(train_loader):
            images = torch.stack(images).to(CFG.device).float()
            bboxes = [target['boxes'].to(CFG.device).float() for target in targets]
            labels = [target['labels'].to(CFG.device).float() for target in targets]
            batch_size = images.shape[0]

            target_res = dict()
            target_res['bbox'] = bboxes
            target_res['cls'] = labels 
            
            # forward pass & calculate loss
            loss = model(images, target_res)['loss']
            loss_value = loss.detach().item()
            train_loss_hist.update(loss_value, batch_size)
            
            # backward
            optimizer.zero_grad() # reset previous gradient
            loss.backward() # backward propagation
            optimizer.step() # parameters update

            # log train loss by step
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.6f}'.format(epoch, CFG.nepochs, step+1, len(train_loader), loss_value))

            del images, targets, bboxes

        # scheduler step
        scheduler.step()

        # Print score after each epoch
        if ((epoch % CFG.print_freq)==0) or (epoch==(CFG.nepochs)):
            val_loss = validation(model, val_loader)
            print ("epoch:[%d] train_loss:[%.6f] val_loss:[%.6f]" % (epoch, train_loss_hist.value, val_loss))

        # wandb.log({
        #     "Train Loss": train_loss,
        #     "Val Loss": val_loss,
        #     "Val mIoU": mIoU,
        #     "Val pix_acc": acc,
        #     "Seg": fig_mask,
        # })

        if early_stopping(model=model, val_loss=val_loss):
            best_metric = {
                'epoch': epoch,
                'val_loss': val_loss
            }
            torch.save(model.state_dict(),
                       os.path.join(CFG.models_path, CFG.model_save_name, f"{CFG.model_save_name}_{str(epoch).zfill(2)}.pt"))

        if early_stopping.early_stop or epoch == CFG.nepochs:
            print("best model information")
            print(f"epoch : {best_metric['epoch']}")
            print(f"val_loss : {best_metric['val_loss']}")
            break

    print ("Done")


def main():
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 개수: {torch.cuda.device_count()}")

    parser = argparse.ArgumentParser(description="Recycle Object Detection")
    parser.add_argument("--config", type=str, default="base_config.json", help=f'train config file (defalut: base_config.json)')
    args = parser.parse_args()

    # parsing config class from custom config.json file
    get_train_config(CFG, os.path.join(CFG.PROJECT_PATH, 'configs', 'train', args.config))

    # initialize wandb settings
    # wandb.init(project="save_the_earth", entity="mignondev")
    # wandb.config.update(CFG.__dict__)

    # make folder structure for training
    make_folder_structure()

    # set every random seed
    seed_everything(CFG.seed)

    # get pytorch data utils (dataset, dataloader)
    train_dataset, val_dataset, train_loader, val_loader = get_data_utils()
    
    # fig = plt.figure(figsize=(12,7))
    # ax = fig.add_subplot(111)
    # print(train_dataset[0][0].numpy().transpose(1,2,0))
    # ax.imshow(train_dataset[0][0].numpy().transpose(1,2,0))
    # plt.savefig("./check.png")

    # get model, optimizer, criterion(not for this task), and scheduler
    model, optimizer, scheduler = get_model()

    # train model
    # validation(model, val_loader) # for test
    train(model, optimizer, scheduler, train_loader, val_loader)

    # finish the logging to wandb
    # wandb.finish()


if __name__ == "__main__":
    main()