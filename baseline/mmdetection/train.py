# Import module

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

#wandb 구축용
import wandb
import plotly

"""
(TODO) 해당 부분에서 CFG로 부르는 것들을 argparse로 바꾸기
"""

# config file 들고오기
cfg = Config.fromfile('./configs/swin/custom_swin_large.py')
cfg.dataset_type = "CocoDataset"

#(TODO) 절대 안 바뀔 상수들이 뭔지 생각하고, 그를 상수처럼 정의하기
IMG_ROOT  = "../../dataset"
TEST_ROOT = "../../dataset"
CLASSES = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


#(TODO) Data preparation config 설정

"""
Data preparation
- data를 train/ valid / test 로 설정하는 부분
- 각 data는 classes / img_prefix / ann_file 등의 옵션이 존재
- 현재의 train.py는 CoCoDataset 기준으로 작성되어있음
- 향후 CustomDataset으로 만들어져 함수를 변경해야 한다면, 이를 바꾸자.
"""

def set_data_prepare_config(cfg: Config) -> None:
    #    무엇을 하더라도, classes와 img_root는 바뀌지 않을 것
    cfg.data.train.classes = CLASSES
    cfg.data.train.img_prefix = IMG_ROOT

    cfg.data.val.classes = CLASSES
    cfg.data.val.img_prefix = IMG_ROOT

    cfg.data.test.classes = CLASSES
    cfg.data.test.img_prefix = IMG_ROOT

    #   TTA 시 bbox 좌표 등이 바뀔 수 있으나, 일단은 이 곳에 고정
    cfg.data.test.ann_file = TEST_ROOT + "test.json"

"""
Data Augmentation

- 현재 Img scaling 부분만 (512,512)로 fix하여 바꿔주고 있음
- 추후 Multi-scale training / validation 시, 이를 가변적으로 바꿔줄 수 있게끔 바꾸기
- Model이 여러 scale로 학습되었다면, 이를 고려해보기
"""


RESIZE = (1024,1024)

def set_augmentation_config(cfg: Config) -> None:
    #   Augmentation은 train-pipeline, val-pipeline, tset-pipeline을 어찌 정의하느냐에 따라 계속 바뀔 수 있음.
    #cfg.data.train.pipeline[2]['img_scale'] = RESIZE # Resize
    cfg.data.val.pipeline[1]['img_scale'] = RESIZE # Resize
    cfg.data.test.pipeline[1]['img_scale'] = RESIZE # Resize
    

"""
Model Config

- Two-stage와 One-stage의 구분을 바꿔주자
- 각 클래스가 공통으로 가지고 있는 config(ex. ROIHead)를 바꿔주고, 함수를 개선시켜 나가나
- 현재의 함수는 Faster RCNN, Cascade RCNN만을 지원
- hasattr 함수를 이용하여 원활한 코딩이 가능하게끔

(TODO) 
[1] One-stage model에도 범용적으로 적용될 수 있는지를 확인할 것
[2] BBoxHead, BboxCoder 등의 여러 인자를 공부해보기 ==> 특정 Model zoo에 꽂혔다면, 그 model zoo의 config를 분석해보기

"""
def set_model_config(cfg: Config) -> None:
    #   현재까지는 Faster RCNN, Cascade RCNN만 되는 것을 확인

    #In case of faster RCNN(baseline) : Dict
    if type(cfg.model.roi_head.bbox_head) == dict:
        cfg.model.roi_head.bbox_head.num_classes = 10

    #In case of cascade RCNN : List[Dict]
    elif type(cfg.model.roi_head.bbox_head) == list:
        for each_head in cfg.model.roi_head.bbox_head:
            if hasattr(each_head, "num_classes"):
                each_head.num_classes = 10 
            else: 
                raise Exception("Num_classes가 없습니다. 제대로 찾으셨나요?")
"""
Training config

- random_seed, samples, single_gpu 등을 지정 가능
- work_dir를 잘 지정하여, 효과적인 실험관리가 가능하게끔 할 수 있음
- optimizer_config, checkpoint_config 확인하기

- runner와 관련된 것들은 대부분 mmcv의 config에 지정된다.

"""

def set_train_config(cfg: Config) -> None:
    cfg.data.samples_per_gpu = 3
    cfg.seed = 2022
    cfg.gpu_ids = [0]
    cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash/swin_large'
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=3)
    cfg.device = get_device()

    #Runner를 통해 epochs를 조절 가능
    cfg.runner.max_epochs = 12

    cfg.resume_from = "work_dirs/faster_rcnn_r50_fpn_1x_trash/from_6_epoch.pth"

    #Metric 지정하는 부분, List로 선언시 list 안의 metric들에 대하여 각각 evaluate하여 터미널 창에다가 보여줌.
    cfg.evaluation.metric = ["bbox"]

"""
Hook config

- TextLoggerHook : 터미널 창에 mAP 지표를 기록시켜줌
- MMDetWandbHook : Wandb와 함께 로깅이 되는 그래프

(TODO) wandb 설치 및 본인의 계정과 연동 + plotly 설치

"""
def set_hook_config(cfg: Config) -> None:
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs={
            'project': 'mmdetection' , 
            "tags" : ["practice_wandb_first"]
            },
            interval=50,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=30,
            bbox_score_thr=0.3),
        ]

"""
(TODO) 아직 이 아래의 baseline은 불완전하므로, 계속 develop하기

Train pipeline

train/val split & CV에 따라 함수를 정의한 후, 그에 따라 유연하게 정의하기
train과 val의 annotation_file이 있는 곳을 지정해주는 곳 (image root는 그대로, annotation file만 나누어져있다고 가정)

- train_default   :: Original baseline(대회에서 주어진 그대로)
- train_val_split :: train/valid json이 나누어져 있다면, 이를 각각 읽어들인다. 
"""

def train_default(cfg: Config) -> None:
    # 아무것도 안한 상태로, 그냥 학습시키는 코드(Baseline)
    train_root = "../../dataset"
    cfg.data.train.ann_file = train_root + "train.json" #train json 정보

def train_val_split(cfg, kfold = False, foldnum= ""):
    # train과 valid가 나뉘어졌다면, 이 함수를 실행시킨다
    if kfold and not foldnum:
        train_root='../../dataset/kfold/'
        cfg.data.train.ann_file = train_root + f'train_fold{foldnum}.json' # train json 정보
        cfg.data.val.ann_file = train_root + f'valid_fold{foldnum}.json' # valid json 정보

    else:
        train_root='../../dataset/groupskfold/'
        cfg.data.train.ann_file = train_root + 'train_fold0.json' # train json 정보
        cfg.data.val.ann_file = train_root + 'valid_fold0.json' # valid json 정보

def set_everything_config(cfg, default = False, train_split = True, kfold = False):
    set_data_prepare_config(cfg)
    set_augmentation_config(cfg)
    set_model_config(cfg)
    set_train_config(cfg)
    set_hook_config(cfg)

    assert int(default) + int(train_split) + int(kfold) ==1 , "Choose only one option: (Default | Single_train_valid | Kfold)"

    #이제 바꿔줘야할 건 Annotation 관련 config & 최종 config들
    if default:
        train_default(cfg)
    
    elif train_split:
        train_val_split(cfg)
    
    elif kfold:
        return cfg
    return cfg


def single_train(cfg):
    set_everything_config(cfg)
    datasets = [build_dataset(cfg.data.train)]
    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

def kfold_train(cfg, fold_num = 5):
    pass
        
if __name__ == "__main__":
    single_train(cfg)

