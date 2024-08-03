# Gender Classifiaction with PETA dataset

## **1.프로젝트 개요**

본 프로젝트 PETA(PEdsTrian Attribute) 데이터셋을 활용하여 성별을 분류하는 모델을 학습,추론,테스트 하는 프로세스를 구축하고 각 실험을 정리하는 로직을 구현하였음.

- 학습 : Pytorch 2.1
- 파라미터 및 관리 도구 : Ray tune, Mlflow

## 2. 환경설정

### 2_1. 설치 방법

#### Case A.Requirements.txt

```
pip install -r requirements.txt

# 이 경우 mlflow 서버를 수동으로 켜야하기에 아래 명령어를 추가로 입력해주세요
./start_mlflow_server.sh
```

#### Case B.Dockerfile [recommend]

**step1**.git clone this repo

```
git clone https://github.com/hajjin2050/Gender_estimation.git
cd Gender_estimation
```

**step2**.Building Docker image

```
docker build -t hajjin_like_spacevision:v1 .
```

**step3.Runnung Docker container**

```
docker run -it --gpus all --ipc=host -v ./:/data --name hajjin_estimation  hajjin_like_spacevision:v1
./start_mlflow_server.sh &
```

## 3.데이터 셋

구축된 학습 데이터 다운로드 : https://drive.google.com/drive/folders/1bK9smyonsrmWmfTBlwVwHPBtpiChTqan?usp=drive_link

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
python tools/train.py --config /workspace/config/config_resnet101.json --data_dir /path/your/dataset
```

#### 구성

- 모델 학습 코드

  - config 폴더에 작성한 json  파일을 활용하여 학습 코드 재사용 용이하게 구축함
  - 각 실험 별로 mlflow에 업로드되는 run_name과 동일한 폴더가 생성되며 아래와 같은 파일들이 아웃풋으로 산출됨

    - config.json : 실험 상세 파라미터
    - confusion_matrix _x.png : 에폭마다 confusion matrix 이미지 저장
    - heatmap_overlay_epoch_x.png : 정성적 평가를 위한 모델의 마지막 레이어 통과할 때  heatmap 을 저장
    - metrics.log : 에폭마다 검증 데이터 셋에 대한 prec,recall,f1-score 기록
    - model_epoch_x.pth : 모델 체크포인트 저장
  - mlflow에도 같은 내용이 올라가지만 버전관리와 백업용도로 로컬('/workspace/runs')에도 저장하고있음
  - 학습 완료 이후 val_f1score를 기준으로 가장 좋은 모델을 선정하여 mlflow에 registration 함.
    -> val_f1score는 기준이며 각 실험마다 Confusion Matrix, metric.log를 분석하여 최고의 모델을 산출함(아직은 수작업)

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
python tools/test.py --config /workspace/config/config_efficientNetB5.json --model /path/to/trained_model.pth --test_dir /path/to/test_dataset_directory
````

#### 구성

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
python tools/inference.py --config /workspace/config/config_efficientNetB5.json --model /path/to/trained_model.pth --input /path/to/input_image_or_directory
```

#### 구성

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
