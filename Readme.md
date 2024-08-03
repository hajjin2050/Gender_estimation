# Gender Classifiaction with PETA dataset

![썸네일 이미지](./data/image.webp)

## **1.프로젝트 개요**

본 프로젝트 PETA(PEdsTrian Attribute) 데이터셋을 활용하여 성별을 분류하는 모델을 학습,추론,테스트 하는 프로세스를 구축하고 각 실험을 정리하고 

## 2. 환경설정

### 2_1. 설치 방법

#### Case A.Requirements.txt

```
pip install -r requirements.txt
```

#### Case B.Dockerfile [recommend]

**step1**.git clone this repo

```
git clone -b main https://github.com/hajjin2050/Gender_estimation.git
cd Gender_estimation
```

**step2**.Building Docker image

```
cd docker
docker build -t hajjin_like_spacevision:v1 .
```

**step3.Runnung Docker container**

```
docker run -it --gpus all --ipc==host -v ./:/data --name hajjin_estimation  hajjin_like_spacevision:v1
./start_mlflow_server.sh &
```

### 2_2.프로젝트 디렉토리 구조

```
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
```

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

## 4.Usage

### 4_1.Train

```
python tools/train.py --config /workspace/config/config_resnet101.json
```

#### 구성

- 모델 학습 데이터
- 학습에 필요한 하이퍼 파라미터, 모델, 데이터 정보,  mlflow 설정을  config 파일로 작성하여 학습을 진행
- 모델의 경우 model 폴더에 구축하여 모델이름을 Config 에 설정
- 학습 중 사용되는 Config는 파라미터 값을 명시하여 checkpoint 와 함께 저장해서 버전관리에 사용함

#### IO(Input/Output)

- Input : Config 경로
- Output : checkpoint, Config, Val log

#### 기능

- 동적 모델 로딩 : 모델 이름을 기반으로 모델을 동적으로 로드
- Ray Tune을 사용한 하이퍼파라미터 튜닝 : Ray Tune을 사용하여 하이퍼파라미터를 튜닝
- MLflow를 사용한 로깅 : 학습 및 검증 지표를 MLflow에 로깅하여, 정량적으로 우수한 모델을 등록함
- Grad-CAM,Confusion Matrix 시각화 :각 실험의 에폭마다  validation 셋에 대하여  Grad-CAM,Confusion Matrix 을 시각화하여 mlflow Artiface에 등록함

### 4_2.Test

````
python test.py --config /workspace/config/config_efficientNetB5.json --model /path/to/trained_model.pth --test_dir /path/to/test_dataset_directory
````

#### 구성

- 모델 테스트 데이터(GT가 있어야 함)
- 구축된 테스트 데이터 셋을 활용하여 모델의 평가지표(Prec, Recall, F1-score)와 각 데이터 추론결과 산출
- 학습떄 사용한 config(['model_name']활용), model,test데이터 경로를 필요로함

#### IO(Input/Output)

- input : config 경로(학습산출물), model 경로( checkpoint), test_dir (테스트 데이터 셋 경로)
- output : 모델 폴더명_test_results.xlsx

#### 기능

- 모델 로드 : 구성 파일과 모델 파일을 사용하여 훈련된 모델 로드
- 데이터 전처리 : 입력 데이터 셋 전처리
- 성능 평가 : 실제 성별과 예측된 성별을 비교하여 Precision, Recall, F1-score 계산
- 결과 저장 각 이미지 GT와 pred를 포함한 결과와 전체 테스트 셋에 대한 성능 지표를 엑셀 파일로 저장

### 4_3.Inference

```
python inference.py --config /workspace/config/config_efficientNetB5.json --model /path/to/trained_model.pth --input /path/to/input_image_or_directory
```

#### 구성

- 모델 추론 데이터(GT없는 추론용 데이터_ demo폴더에 있음)
- 훈련된 모델을 사용하여 새로운 이미지 또는 이미지 디렉토리에 대해 예측 수행
- 학습 때 사용된 config,model, 입력이미지 경로를 필요로 함

#### IO(Input/Output)

- input : config경로(학습산출물), model 경로(checkpoint), input(단일 이미지 또는 이미지가 포함된 디렉토리 경로)
- output : 모델 폴더명_inference_results.xlsx

#### 기능

- 모델로드 : 구성 파일과 모델 파일 사용하여 로드
- 데이터 전처리 : 입력 이미지 또는 디렉토리 내의 이미지 전처리
- 추론 수행 : 모델을 사용하여 각 이미지에 대한 추론 수행
- 결과 저장 : 각 이미지에 대한 예측 결과를 포함한 결과를 엑셀 파일로 저장

## 5.Dependencies

이 프로젝트는 다음과 같은 주요 라이브러리에 의존합니다.

- Python 3.x
- Pytorch >2.0
- MLflow
- Ray Tune
- pandas
- scikit-learn
- Pillow
