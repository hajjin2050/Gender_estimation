# Gender Classifiaction with PETA dataset

## **1.프로젝트 개요**

본 프로젝튼느 PETA(PEdsTrian Attribute) 데이터셋을 활용하여 성별을 분류하는 모델을 구축하고, Ray Tune 을 사용하여 하이퍼파라미터 튜닝을 자동화하는 것을 목표로 하였고, 기존에 널리 사용되는 분류모델을 기반으로 정량적인 지표로는 Prec,Recall,F1-score,CM(confusion Matrix), 정성적인 지표로는 Grad-CAM을 활용하여 모델 예측 근거를 시각화 하였음.

## 2. 환경설정

### 2_1. 설치 방법

```
pip install -r requirements.txt
```

### 2_2.프로젝트 디렉토리 구조

workspace/
├── config/
│   ├── config_efficientNetB5.json
│   ├── config_resnet101.json
│   └── config_densenet121.json
├── data/
│   ├── dataloader.py
│   ├── dataset.py
│   ├── dataset_source.py
│   ├── grad_cam.py

│   ├── scheme.py

│   └── PETA_gender_classification_dataset_v1/
├── model/
│   ├── base_model.py
│   ├── CustomEfficientNetB5.py
│   ├── resnet101.py
│   └── densenet121.py
├── tools/
│   ├── train_loop.py
│   └── utils.py
│   └── train.py
└── requirements.txt

## 3.데이터 셋

* **Train Dataset** :

  data/PETA_gender_classification_train.csv 참조

  ```
  image_id,gender
  0001.jpg,0
  0002.jpg,1
  ...
  ```
* **Valid Dataset** :

  data/PETA_gender_classification_.csv 참조

  ```
  image_id,gender
  1001.jpg,0
  1002.jpg,1
  ...
  ```

## 학습 방
