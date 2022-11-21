"""
(TODO) 다음과 같은 방식으로 데이터셋을 구현해보기

1. Random index로 split data
2. Cross-validation split
"""
import numpy as np
import json 
import os 
import random 
from pycocotools.coco import COCO
import copy
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold


"""
Type 1: random_split by index
"""

#Ref : https://dacon.io/competitions/official/235672/codeshare/1773
#Ref2: https://www.kaggle.com/code/nekokiku/8th-training-mmdetection-cascadercnn-weight-bias


CATEGORIES = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]



def get_counter(path, isfold = False):
    """
    train_json과 val_json의 Counter를 비교하여, 분포를 비교한다.
    """
    train_json_path = os.path.join(path, "train.json") if not isfold else os.path.join(path, "train_fold0.json")
    val_json_path = os.path.join(path, "val.json") if not isfold else os.path.join(path, "val_fold0.json")
    with open(train_json_path, "rb") as f:
        train_json = json.load(f)
    with open(val_json_path, "rb") as f:
        val_json = json.load(f)

    train_labels = [ann["category_id"] for ann in train_json["annotations"]]
    val_labels = [ann["category_id"] for ann in val_json["annotations"]]    
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    train_counter = {i:train_counter[i] for i in range(10)}
    val_counter = {i:val_counter[i] for i in range(10)}

    print("train_counter:", train_counter)
    print("val_counter:", val_counter)
    print("length of train:", sum(train_counter.values()))
    print("length of val:", sum(val_counter.values()))

    train_summation = sum(train_counter.values())
    val_summation = sum(val_counter.values())

    print("Train:", [f"{(i/train_summation) * 100}%" for i in train_counter.values()])
    print("Valid:", [f"{(i/val_summation) * 100}%" for i in val_counter.values()])

    return train_counter, val_counter

def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())
    print([f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)])

def random_split_dataset(coco_json, categories, test_data_num = 900, random_seed = 2022):
    new_coco = {}

    cat_ids = coco_json.getCatIds(categories)
    train_img_ids = set()
    test_img_ids = set()

    for cat in cat_ids[::-1]:
        img_ids = copy.copy(coco_json.getImgIds(catIds=[cat]))
        random.shuffle(img_ids)
        tn = min(test_data_num, int(len(img_ids) * 0.2))
        new_test = set(img_ids[:tn])
        exist_test_ids = new_test.intersection(train_img_ids)
        test_ids = new_test.difference(exist_test_ids)
        train_ids = set(img_ids).difference(test_ids)
        print(tn, len(img_ids), len(new_test), len(test_ids), len(train_ids))
        train_img_ids.update(train_ids)
        test_img_ids.update(test_ids)
#        print(len(test_img_ids))

    # prune duplicates
    dup = train_img_ids.intersection(test_img_ids)
    train_img_ids = train_img_ids - dup

    print("train_img_ids:", train_img_ids)
    print("test_img_ids:", test_img_ids)

    train_anno_ids = set()
    test_anno_ids = set()
    for cat in cat_ids:
        train_anno_ids.update(coco_json.getAnnIds(imgIds=list(train_img_ids), catIds=[cat]))
        test_anno_ids.update(coco_json.getAnnIds(imgIds=list(test_img_ids), catIds=[cat]))

    assert len(train_img_ids.intersection(test_img_ids)) == 0, 'img id conflicts, {} '.format(train_img_ids.intersection(test_img_ids))
    assert len(train_anno_ids.intersection(test_anno_ids)) == 0, 'anno id conflicts'
    print('train img ids #:', len(train_img_ids), 'train anno #:', len(train_anno_ids))
    print('valid img ids #:', len(test_img_ids), 'test anno #:', len(test_anno_ids))
    new_coco_test = copy.deepcopy(new_coco)

    new_coco["images"] = coco_json.loadImgs(list(train_img_ids))
    new_coco["annotations"] = coco_json.loadAnns(list(train_anno_ids))
    new_coco["categories"] = coco_json.loadCats(cat_ids)

    new_coco_test["images"] = coco_json.loadImgs(list(test_img_ids))
    new_coco_test["annotations"] = coco_json.loadAnns(list(test_anno_ids))
    new_coco_test["categories"] = coco_json.loadCats(cat_ids)

    print('new train split, images:', len(new_coco["images"]), 'annos:', len(new_coco["annotations"]))
    print('new valid split, images:', len(new_coco_test["images"]), 'annos:', len(new_coco_test["annotations"]))

    output_dir = "./random_split"

    with open(f'./{output_dir}/train.json', 'w') as f:
        json.dump(new_coco, f)
    with open(f'./{output_dir}/valid.json', 'w') as f:
        json.dump(new_coco_test, f)

    

################
# SPLIT DATA   #
################

def skfold_group_dataset(coco_json, categories, random_seed = 2022):

    #Make folder first
    output_dir = "./groupskfold"
    os.makedirs(output_dir, exist_ok = True)

    new_coco = {}
    cat_ids = coco_json.getCatIds(categories)
    with open("./train.json", "rb") as f: 
        data = json.load(f)
    
    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
    X = np.ones((len(data['annotations']),1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_img_ids = set(groups[train_idx])
        test_img_ids = set(groups[val_idx])

        get_distribution(y[train_idx])
        get_distribution(y[val_idx])

        train_anno_ids = set()
        test_anno_ids = set()
        for cat in cat_ids:
            train_anno_ids.update(coco_json.getAnnIds(imgIds=list(train_img_ids), catIds=[cat]))
            test_anno_ids.update(coco_json.getAnnIds(imgIds=list(test_img_ids), catIds=[cat]))

        assert len(train_img_ids.intersection(test_img_ids)) == 0, 'img id conflicts, {} '.format(train_img_ids.intersection(test_img_ids))
        assert len(train_anno_ids.intersection(test_anno_ids)) == 0, 'anno id conflicts'
        print('train img ids #:', len(train_img_ids), 'train anno #:', len(train_anno_ids))
        print('valid img ids #:', len(test_img_ids), 'test anno #:', len(test_anno_ids))
        new_coco_test = copy.deepcopy(new_coco)

        new_coco["images"] = coco_json.loadImgs(list(train_img_ids))
        new_coco["annotations"] = coco_json.loadAnns(list(train_anno_ids))
        new_coco["categories"] = coco_json.loadCats(cat_ids)

        new_coco_test["images"] = coco_json.loadImgs(list(test_img_ids))
        new_coco_test["annotations"] = coco_json.loadAnns(list(test_anno_ids))
        new_coco_test["categories"] = coco_json.loadCats(cat_ids)

        print('new train split, images:', len(new_coco["images"]), 'annos:', len(new_coco["annotations"]))
        print('new valid split, images:', len(new_coco_test["images"]), 'annos:', len(new_coco_test["annotations"]))


        train_output_dir = f"{output_dir}/train_fold{fold}.json"
        val_output_dir = f"{output_dir}/valid_fold{fold}.json"

        with open(train_output_dir, "w") as f:
            json.dump(new_coco, f)
        with open(val_output_dir, "w") as f:
            json.dump(new_coco_test, f)
        
        print(f"Fold {fold} json complete")


RANDOM_SPLIT = False
SKGFOLD = True

if RANDOM_SPLIT:  
    coco = COCO('./train.json')
    random_split_dataset(coco,CATEGORIES )
    
if SKGFOLD:  
    coco = COCO('./train.json')
    skfold_group_dataset(coco, CATEGORIES)

