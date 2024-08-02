import os
import shutil
import random
from pathlib import Path

def move_data_to_validation(data_dir, val_split=0.05):
    """
    Moves a specified percentage of training data to the validation folder.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory.
    val_split : float, optional
        Percentage of training data to move to validation (default is 0.05).
    """
    train_dir = os.path.join(data_dir, '01.train')
    val_dir = os.path.join(data_dir, '02.valid')

    # Ensure validation directory exists
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    for class_name in os.listdir(train_dir):
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)

        # Ensure class directory in validation exists
        Path(class_val_dir).mkdir(parents=True, exist_ok=True)

        # Get all files in the class training directory
        files = os.listdir(class_train_dir)
        num_files_to_move = int(len(files) * val_split)

        # Randomly select files to move
        files_to_move = random.sample(files, num_files_to_move)

        for file_name in files_to_move:
            src_file = os.path.join(class_train_dir, file_name)
            dest_file = os.path.join(class_val_dir, file_name)
            shutil.move(src_file, dest_file)

    print(f"Moved {val_split*100}% of training data to validation folder.")

# 사용 예시
data_dir = '/workspace/PM_source/work/PETA_gender_classification_dataset'  # 데이터셋 디렉토리 경로
move_data_to_validation(data_dir, val_split=0.05)
