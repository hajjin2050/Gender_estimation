import os
import pandas as pd

def extract_gender_from_path(file_path: str) -> str:
    """
    파일 경로에서 성별 정보를 추출합니다.

    Parameters
    ----------
    file_path : str
        이미지 파일의 경로.

    Returns
    -------
    str
        추출된 성별 정보 ("male" 또는 "female").
    """
    if "00female" in file_path:
        return "female"
    elif "01male" in file_path:
        return "male"
    else:
        return "unknown"

def create_csv(data_dir: str, output_csv: str) -> None:
    """
    이미지 파일 경로를 기반으로 CSV 파일을 생성합니다.

    Parameters
    ----------
    data_dir : str
        이미지 파일이 저장된 디렉토리의 최상위 경로.
    output_csv : str
        생성된 CSV 파일의 경로.

    Returns
    -------
    None
    """
    data = {"image_id": [], "gender": []}

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):  # 이미지 파일 확장자를 확인합니다.
                gender = extract_gender_from_path(root)
                data["image_id"].append(file)  # 전체 경로가 아닌 파일 이름만 추가합니다.
                data["gender"].append(gender)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV 파일이 {output_csv}에 저장되었습니다.")

# 사용 예시
train_data_dir = "/workspace/PETA_gender_classification_dataset_v1/train"
val_data_dir = "/workspace/PETA_gender_classification_dataset_v1/valid"
train_csv_path = "/workspace/PETA_gender_classification_train.csv"
val_csv_path = "/workspace/PETA_gender_classification_val.csv"

create_csv(train_data_dir, train_csv_path)
create_csv(val_data_dir, val_csv_path)