# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
import torch
import os, random, pprint
import json


def get_train_config(CFG, config):
    '''
        train config file parser
    '''

    with open(config) as f:
        config = json.load(f)
    
    # environment_parameters
    CFG.coco_train_json = config["environment_parameters"]["coco_train_json"]
    CFG.coco_val_json = config["environment_parameters"]["coco_val_json"]

    # hyper_parameters
    CFG.learning_rate = config["hyper_parameters"]["learning_rate"]
    CFG.train_batch_size = config["hyper_parameters"]["train_batch_size"]
    CFG.valid_batch_size = config["hyper_parameters"]["valid_batch_size"]
    CFG.nepochs = config["hyper_parameters"]["nepochs"]
    CFG.patience = config["hyper_parameters"]["patience"]
    CFG.seed = config["hyper_parameters"]["seed"]
    CFG.num_workers = config["hyper_parameters"]["num_workers"]

    # network_env
    CFG.model = config["network_env"]["model"]
    CFG.optimizer = config["network_env"]["optimizer"]
    CFG.optimizer_params = config["network_env"]["optimizer_params"]
    CFG.scheduler = config["network_env"]["scheduler"]
    CFG.scheduler_params = config["network_env"]["scheduler_params"]
    CFG.train_augmentation = config["network_env"]["train_augmentation"]
    CFG.val_augmentation = config["network_env"]["val_augmentation"]
    CFG.model_save_name = config["network_env"]["model_save_name"]

    CFG.coco_train_json = os.path.join(CFG.BASE_DATA_PATH, CFG.coco_train_json)
    CFG.coco_val_json = os.path.join(CFG.BASE_DATA_PATH, CFG.coco_val_json)
    CFG.docs_path = os.path.join(CFG.PROJECT_PATH, CFG.docs_path)
    CFG.models_path = os.path.join(CFG.PROJECT_PATH, CFG.models_path)

    # pprint.pprint(CFG.__dict__)


def get_test_config(CFG, config):
    '''
        test config file parser
    '''

    with open(config) as f:
        config = json.load(f)
    
    # environment_parameters
    CFG.coco_test_json = config["environment_parameters"]["coco_test_json"]

    # hyper_parameters
    CFG.batch_size = config["test_parameters"]["batch_size"]
    CFG.seed = config["test_parameters"]["seed"]
    CFG.num_workers = config["test_parameters"]["num_workers"]
    CFG.score_threshold = config["test_parameters"]["score_threshold"]

    # network_env
    CFG.model = config["network_env"]["model"]
    CFG.test_augmentation = config["network_env"]["test_augmentation"]
    CFG.model_name = config["network_env"]["model_name"]

    CFG.coco_test_json = os.path.join(CFG.BASE_DATA_PATH, CFG.coco_test_json)
    CFG.docs_path = os.path.join(CFG.PROJECT_PATH, CFG.docs_path)
    CFG.models_path = os.path.join(CFG.PROJECT_PATH, CFG.models_path)

    # pprint.pprint(CFG.__dict__)

# 시드 고정 함수
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.count = 0

    def update(self, value, batch_size):
        self.current_total += value * batch_size
        self.count += batch_size

    @property
    def value(self):
        if self.count == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.count

    def reset(self):
        self.current_total = 0.0
        self.count = 0


# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.best_metric = None

    def __call__(self, model, val_loss=None, metric_score=None):
        save_flag = True
        if val_loss:
            score = -val_loss
        
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint_loss(val_loss, model)

            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                save_flag = False

            else:
                self.best_score = score
                self.save_checkpoint_loss(val_loss, model)
                self.counter = 0


        elif metric_score:
            score = metric_score

            if self.best_score is None:
                self.best_score = np.inf
                self.save_checkpoint_score(score, model)
                
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                save_flag = False
                
            else:
                self.save_checkpoint_score(score, model)
                self.counter = 0

        return save_flag

    def save_checkpoint_loss(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


    def save_checkpoint_score(self, score, model):
        '''Saves model when metric_score increase.'''
        if self.verbose:
            self.trace_func(f'Validation score increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        self.best_score = score