import sys
sys.path.append("/workspace")
import os
import json
import argparse
import importlib
from datetime import datetime
import uuid

import mlflow.models.signature
import torch
import torch.optim as optim
import mlflow
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import mlflow 

from data.dataloader import get_data_loaders
from data.grad_cam import GradCAM
from data.dataset import CustomDataset
from data.dataset_source import DatasetSource
from tools.train_loop import train_one_epoch, validate
from tools.utils import save_confusion_matrix, log_mlflow_params, log_mlflow_metrics, log_mlflow_images

# 모델 클래스 로드 함수
def load_model_class(model_name : str) -> type:
    """
    모델 모듈에서 동적 모델 클래스를 로드
    -------------
    input : model_name(Config['model_name']) 
    output : model_class(timm활용)
    --------------
    """
    module_name = f"model.{model_name}"
    class_name = model_name
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    print(model_class)
    return model_class

# Ray Tune 설정 생성 함수
def create_tune_config(tune_params : dict) -> dict:
    """
    제공된 tune_params로 부터 Ray Tune 구성을 생성
    -------------
    input : tune_params(ray tune 매개변수 포함)
    output : ray tune 구성 딕셔너리
    -------------

    """
    tune_config = {}
    for param_name, param_values in tune_params.items():
        if param_values["type"] == "loguniform":
            tune_config[param_name] = tune.loguniform(param_values["min"], param_values["max"])
        elif param_values["type"] == "uniform":
            tune_config[param_name] = tune.uniform(param_values["min"], param_values["max"])
        elif param_values["type"] == "choice":
            tune_config[param_name] = tune.choice(param_values["values"])
    return tune_config

# Config 파일 로드 함수
def load_config(config_path : str) -> dict:
    """
    구축해놓은 모델, tune 파라미터 등을 입력한 Config 파일에서 정보를 가져옴
    -------------
    input : Config 경로
    output : dict 
    -------------

    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# 데이터 셋 로깅 함수
def log_datasets(train_csv_path, val_csv_path):
    """
    Train 및 Val 데이터 MLflow 로깅
    """
    train_dataset_source = DatasetSource(train_csv_path)
    train_dataset = CustomDataset(source=train_dataset_source, name="PETA Gender Classification Train Dataset", dataset_type='train')
    mlflow.log_param("train_dataset_name", train_dataset.name)
    mlflow.log_param("train_dataset_digest", train_dataset.digest)
    mlflow.set_tag("train_dataset_schema", train_dataset.schema)

    val_dataset_source = DatasetSource(val_csv_path)
    val_dataset = CustomDataset(source=val_dataset_source, name="PETA Gender Classification Val Dataset", dataset_type='val')
    mlflow.log_param("val_dataset_name", val_dataset.name)
    mlflow.log_param("val_dataset_digest", val_dataset.digest)
    mlflow.set_tag("val_dataset_schema", val_dataset.schema)
    
# 하이퍼 파라미터 로깅 함수
def log_hyperparameters(config):
    """
    하이퍼파라미터 MLflow 로깅
    """
    mlflow.log_params({
        "model_architecture": config["model_name"],
        "total_epochs": config["epochs"],
        "learning_rate": config["lr"],
        "drop_rate": config["drop_rate"],
        "batch_size": config["batch_size"],
        "weight_decay": config["weight_decay"]
    })
    log_mlflow_params(config)

# 학습지표 로깅 함수
def log_metrics(metrics, step):
    """
    학습지표 MLflow 로깅.
    """
    log_mlflow_metrics(metrics, step=step)

# 최적의 모델 저장 함수
def save_best_model(model, optimizer, val_f1, best_val_f1, epoch, run_dir):
    """
    최적의 모델을 저장(valid f1-score 기준)
    """
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_path = f"{run_dir}/model_epoch_{epoch+1}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, best_model_path)
        return best_val_f1, best_model_path
    return best_val_f1, None

# 모델 로깅 및 등록 함수
def log_and_register_model(model, run_id, run_name, signature, input_example, artifact_paths):
    """
    모델을 MLflow에 로깅하고 등록.
    """
    mlflow.pytorch.log_model(model, "model", registered_model_name=f"Best_Model_{run_name}",
                            description=f"This model is a {run_name} trained on the PETA dataset.",
                            tags={"model_architecture": run_name.split('_')[0], "dataset": "PETA_v1", "run_name": run_name},
                            signature=signature,
                            input_example=input_example.cpu().numpy())
    for path in artifact_paths:
        mlflow.log_artifact(path)
    mlflow.register_model(f"runs:/{run_id}/model", "Best_Gender_Classifier_Model")

# 모델 학습 및 검증 함수
def train_and_validate(config: dict, checkpoint_dir: str = None) -> None:
    """
    모델 학습 & 검증.
    Config 구성에 따라 학습, 검증 루프를 거쳐 지정한 학습 지표에 따라 MLflow에 결과 등록.
    """
    # 모델 로드
    model_class = load_model_class(config["model_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=2, pretrained=True, drop_rate=config["drop_rate"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    # 데이터 로더 선언
    train_csv_path = config["train_csv_path"]
    val_csv_path = config["val_csv_path"]
    image_root_dir = config["image_root_dir"]
    train_loader, val_loader = get_data_loaders(train_csv_path, val_csv_path, image_root_dir, batch_size=config["batch_size"])

    # 체크포인트 로드
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # MLflow tracking 설정
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    grad_cam = GradCAM(model, model.model.conv_head)
    best_val_f1 = 0
    best_model_path = None
    run_name = f"{config['model_name']}_dropout={config['drop_rate']}_lr={config['lr']}"

    try:
        with mlflow.start_run(run_name=run_name) as run:
            input_example = next(iter(val_loader))[0].to(device)
            input_example = input_example[:1]  # 하나의 배치만 사용
            output_example = model(input_example).detach().cpu().numpy()
            signature = mlflow.models.signature.infer_signature(input_example.cpu().numpy(), output_example)

            # 데이터셋 로깅
            log_datasets(train_csv_path, val_csv_path)
            log_hyperparameters(config)

            artifact_paths = []

            for epoch in range(config['epochs']):
                # 학습
                train_loss, train_precision, train_recall, train_f1 = train_one_epoch(model, device, train_loader, optimizer)
                # 검증
                val_loss, val_precision, val_recall, val_f1, cm_buf = validate(model, device, val_loader)

                # 메트릭 계산
                metrics = {
                    "train_loss": train_loss,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1
                }

                # 메트릭 로깅
                log_metrics(metrics, epoch)

                # 최적의 모델 저장
                best_val_f1, best_model_path = save_best_model(model, optimizer, val_f1, best_val_f1, epoch, "runs")

                # 혼동 행렬 및 이미지 로깅
                cm_filename = save_confusion_matrix(cm_buf, "runs", epoch)
                mlflow.log_artifact(cm_filename)
                artifact_paths.append(cm_filename)
                
                inputs, targets = next(iter(val_loader))
                inputs = inputs.to(device)
                targets = targets.to(device)
                log_mlflow_images(inputs, model, grad_cam, "runs", epoch)

                print(f"Epoch {epoch+1}/{config['epochs']} - "
                      f"Train Loss: {train_loss:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f} "
                      f"Val Loss: {val_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

            if best_model_path:
                # 최적의 모델 로깅 및 등록
                log_and_register_model(model, run.info.run_id, run_name, signature, input_example, artifact_paths)

    except Exception as e:
        print(f"Exception occurred: {e}")
        mlflow.end_run(status="FAILED")
        raise

# Trainable class
class Trainable(tune.Trainable):
    def setup(self, config:dict) -> None :
        """
        모델, 데이터 로더 및 기타 설정 초기화
        -------------
        input : config 
        output : None
        -------------
        """
        # 학습에 필요한 설정 초기화
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_class = load_model_class(config["model_name"])
        self.model = model_class(num_classes=2, pretrained=True, drop_rate=config["drop_rate"]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        
        # 현재 시간과 UUID를 사용하여 실행 이름 생성
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        self.run_name = f"{config['model_name']}_{current_time}_{unique_id}"
        
        # 데이터 로더 초기화
        self.train_loader, self.val_loader = get_data_loaders(data_dir=args.data_dir, batch_size=config["batch_size"])
        
        # MLflow 설정
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        
        # 기타 초기화
        self.run = None
        self.grad_cam = GradCAM(self.model, self.model.grad_cam_layer)
        self.run_dir = f"/workspace/Gender_estimation/runs/{self.run_name}"
        os.makedirs(self.run_dir, exist_ok=True)
        self.max_epochs = config["epochs"]

        # 설정 파일 저장
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # 모델 트랙킹을 위한 변수 초기화
        self.best_val_f1 = 0
        self.best_model_path = None
        self.patience = 5
        self.no_improvement_epochs = 0
    
    # 파라미터 로깅 함수
    def log_hyperparameters(self):
        """
        하이퍼파라미터를 MLflow에 로깅합니다.
        """
        mlflow.log_params({
            "model_architecture": self.config["model_name"],
            "total_epochs": self.config["epochs"],
            "learning_rate": self.config["lr"],
            "drop_rate": self.config["drop_rate"],
            "batch_size": self.config["batch_size"],
            "weight_decay": self.config["weight_decay"]
        })
    
    # 최적 모델 저장 함수
    def save_best_model(self, val_f1):
        """
        validation 기준 f1-score 갱신 시 저장.
        """
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_model_path = os.path.join(self.run_dir, f"model_epoch_{self.iteration+1}.pth")
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, self.best_model_path)
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1  
             
    # 모델 로드 함수
    def load_best_model(self):
        """
        저장된 최적의 모델을 로드합니다.
        """
        if self.best_model_path:
            self.model.load_state_dict(torch.load(self.best_model_path)["model_state_dict"])
    
    # 성능 지표, CM  mlflow 서버 전송        
    def log_metrics_and_artifacts(self, metrics, cm_buf):
        """
        메트릭,CM 등의 아티팩트를 MLflow에 로깅.
        """
        log_mlflow_metrics(metrics, step=self.iteration)
        cm_filename = save_confusion_matrix(cm_buf, self.run_dir, self.iteration)
        mlflow.log_artifact(cm_filename, artifact_path="confusion_matrix")

        inputs, targets = next(iter(self.val_loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        log_mlflow_images(inputs, self.model, self.grad_cam, self.run_dir, self.iteration)

        log_path = os.path.join(self.run_dir, "metrics.log")
        with open(log_path, 'a') as log_file:
            log_file.write(f"Epoch {self.iteration+1} - " + json.dumps(metrics) + "\n")      
                  
    # 최적 모델 mlflow model registration
    def register_model(self):
        """
        최적의 모델을 MLflow에 등록.
        """
        self.load_best_model()
        mlflow.pytorch.log_model(self.model, "model", registered_model_name="Best_Gender_Classifier_Model",
                                 description="This model is a CustomEfficientNetB5 trained on the PETA dataset.",
                                 tags={"model_architecture": self.config["model_name"], "dataset": "PETA_v1", "run_name": self.run_name})
        mlflow.register_model(f"runs:/{self.run_id}/model", "Best_Gender_Classifier_Model")
        

    # 학습 및 검증 함수
    def step(self) -> dict:
        """
        학습 및 검증 단계를 실행합니다.
        -------------
        input : 
        output : 학습 및 검증 메트릭 포함하는 dict 데이터 
        -------------
        """
        if self.run is None:
            self.run = mlflow.start_run(run_name=self.run_name)
            self.run_id = self.run.info.run_id
            mlflow.set_tag("dataset", "PETA_gender_classification_dataset_v1")
            self.log_hyperparameters()

        try:
            # 한 에포크 동안 학습
            train_loss, train_precision, train_recall, train_f1 = train_one_epoch(self.model, self.device, self.train_loader, self.optimizer)
            # 검증
            val_loss, val_precision, val_recall, val_f1, cm_buf = validate(self.model, self.device, self.val_loader)

            # 메트릭 계산
            metrics = {
                "train_loss": train_loss,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            }

            # 최적의 모델 저장
            self.save_best_model(val_f1)
            # 메트릭 및 아티팩트 로깅
            self.log_metrics_and_artifacts(metrics, cm_buf)

            # 현재 에포크 정보 출력
            print(f"Epoch {self.iteration+1}/{self.config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f} "
                  f"Val Loss: {val_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

            # 조기 종료 조건 확인
            if self.no_improvement_epochs >= self.patience:
                print("조기 종료 조건에 도달했습니다.")
                self.register_model()
                mlflow.end_run()
                return {"done": True, "best_model_path": self.best_model_path}

            return {**metrics, "best_model_path": self.best_model_path}

        except Exception as e:
            print(f"Exception occurred: {e}")
            mlflow.end_run(status="FAILED")
            raise e
        
    def cleanup(self):
        """
        MLflow 실행 종료.
        """
        if self.run is not None:
            mlflow.end_run()

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """
        체크포인트 저장.
        -------------
        input : checkpoint_dir 체크포인트 저장할 경로
        output : 지정된 체크포인트 경로 
        -------------
        """
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        체크포인트 로드.
        -------------
        input : checkpoint_path 로드할 체크포인트의 경로
        output : None
        -------------
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with Ray Tune")
    parser.add_argument("--data_dir", type=str, help="Path to inlcuded Trian/Valid folder")
    parser.add_argument("--config", type=str, help="Path to the JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    tune_config = create_tune_config(config["tune_params"])
    tune_config["model_name"] = config["model_name"]
    tune_config["mlflow"] = config["mlflow"]

    scheduler = ASHAScheduler(
        max_t=config["tune_params"]["epochs"]["values"][0],
        grace_period=1,
        reduction_factor=2
    )


    analysis = tune.run(
        Trainable,
        resources_per_trial={"cpu": 4, "gpu": 2},
        config=tune_config,
        metric="val_f1",
        mode="max",
        num_samples=10,
        scheduler=scheduler,
        name="tune_gender_classifier"
    )
    
    best_trial = analysis.get_best_trial(metric="val_f1", mode="max")
    best_model_path = best_trial.last_result["best_model_path"]
    best_run_dir = os.path.dirname(best_model_path)
    print(f"Best run directory: {best_run_dir}")
    print(f"Best config: {best_trial.config}")