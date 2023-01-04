<div align=center>

![header](https://capsule-render.vercel.app/api?type=waving&text=재활용쓰레기%20검출%20대회&color=7F7FD5&fontColor=FFFFFF&fontSize=50&height=200)

</div> 

# 💘 CV-02조 비전로켓단

<div align=center>

|<img src="https://user-images.githubusercontent.com/72690566/200118081-7f8e4279-04ef-4269-abde-80b9ea89e87a.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118119-d21769d2-ff0d-4e15-9e6d-aa863e700f36.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118141-2de150f1-98cb-4cbd-8ce8-419c1ebb0678.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118162-f25ae93e-18c1-462f-8298-c6ff5c95ee79.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118175-ba5859db-5a2f-4457-a8e2-878f8cc1140e.png" width="80">|
|:---:|:---:|:---:|:---:|:---:|
|구상모|배서현|이영진|권규보|오한별|
|T4008|T4095|T4155|T4011|T4128|

</div>

# Result
- Public 4등 -> Private 4등
![image](https://user-images.githubusercontent.com/81371318/210516357-69bf1b27-7794-4417-be62-5b75d56dae96.png)


### 사진을 입력받아, 재활용 쓰레기의 종류와 위치를 검출하는 모델


# 🌳 Folder Structure
```
```


# ❓ 프로젝트 개요

## 1. Task 소개

수많은 쓰레기들이 버려지고 있는데, 이 중에서는 잘못 버려지고 있는 쓰레기들이 상당하다.
본 대회는, 주어진 이미지로부터 재활용의 쓰레기 종류와 위치를 검출하는 모델이다.
구체적으로는, 쓰레기 이미지 데이터와 COCO format으로 된 bbox 정보가 주어질 때, **쓰레기의 종류(10개 중 1개)와 그 위치를 pascal-voc 포맷으로 제출한다.**

## 2. 작업 환경

- 컴퓨팅 환경 : V100 GPU
- 협업 도구 : Notion, Slack, Wandb, GitHub

## 3. 작업의 순서

<div align=center>

<img src="https://user-images.githubusercontent.com/72690566/200120015-b52eb581-764f-41b0-80fe-b083d9accd0f.png">

</div>
  
강의자료에 주어진 Workflow를 참고하여, 프로젝트 타임라인을 위와 같이 설정하였다.

# ❇️ 프로젝트 팀 구성 및 역할

<div align=center>

|전체|문제 정의, 계획 및 타임라인 수립, 모델 튜닝, 아이디어 제시|
|:----------:|:------:|
|구상모 &nbsp;&nbsp;&nbsp;&nbsp;|CV 전략 수립, wandb 연동, multi-scale training 실험, swin계열 모델 실험, 모델 디버깅|
|권규보|EDA , YOLOv7 실험, mosaic/mixup 실험, data augmentation, background dataset 추가, TTA 도입|
|배서현|Baseline code 성능 실험, backbone/optimizer 실험, hyperparameter tuning, model ensemble(WBF)|
|오한별|기본적인 모델 성능 실험, custom-dataset 제작, 1-stage model 실험, YOLOv7실험, preprocessing 실험, Dataset 추가 실험, Ensemble 실험|
|이영진|TOOD 모델 도입, Github 관리|

</div>
</div>

# **🔑 문제 정의**

## 1. EDA and Model debugging

- EDA결과, 다음과 같은 문제점을 확인했다.

### Problem 1.  Data imbalance

![image](https://user-images.githubusercontent.com/81371318/210517895-4fc15ed4-4a8a-400c-8c29-67fa9c10b4d0.png)

- 극심한 data imbalance가 관찰된다. 가장 많은 Paper는 4천개, 가장 적은 Battery는 150여장 있다.

- **Approach 1.** 적은 Label을 제대로 탐지하지 못한다면, 이를 본 대회의 Key problem으로 간주하자.

- **Approach 2.** Model debugging의 일환으로, **confusion matrix**를 통해 모델의 성능을 디버깅해보자.

- **Approach 3.** **TIDE라는 모델 분석 프레임워크**를 통해, 우리 모델이 어떤 것에 제일 취약한지 디버깅해보자.


### Problem 2. Classification is harder than Region proposal

- Confusion matrix와 TIDE를 통해 본 결과, 우리 모델이 클래스 구분을 잘 못하고 있단 사실을 깨달았다.

- **Approach 4.** backbone 실험을 하여, 좋은 backbone을 찾아 이미지의 semantic feature를 찾을 수 있도록 하자.
- **Approach 5.** 다양한 data 실험을 하자.


# ✍️ Modeling

<div align=center>

|모델 구조    |특징 및 관찰 | LB mAP|
|:----------:|:------:|:------:|
|Faster RCNN + ResNet 계열(R50 -> R101 -> R152)| backbone이 커질수록 mAP 증가(confusion matrix, TIDE 동일) | 0.411 ->0.418 -> 0.438 |
|Cascade R-CNN + Swin 계열(Swin-T, Swin-L) | Swin계열의 시도 | 0.518 ->0.54 |
|Cascade R-CNN + SwinL + Data aug | Brightness, geometry 계열 시도 | 0.57|
|위와 동일 + Multi-scale | multi-scale training 적용 | 0.61|
|위와 동일 + 5-fold | multi-scale training 적용 | 0.67|
|Yolo-v7 + 5-fold | 모델 다양성을 위해 1-stage 계열 이용 | 0.65|
|TOOD + 5-fold | 모델 다양성을 위해 TOOD 이용 | 0.67|
|Final ensemble | Cascade : TOOD : YOLO = 1: 1: 1 | 0.7|

</div>

</div>

<div align=center>  

![Footer](https://capsule-render.vercel.app/api?type=waving&color=7F7FD5&fontColor=FFFFFF&height=200&section=footer)

</div>
