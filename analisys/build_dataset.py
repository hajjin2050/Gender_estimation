import os
import shutil
import glob
from tqdm import tqdm

def parse_label_file(label_file):
    gender_labels = {}
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            index = parts[0]
            attributes = parts[1:]
            for attr in attributes:
                if attr == 'personalMale':
                    gender_labels[index] = '00.male'
                elif attr == 'personalFemale':
                    gender_labels[index] = '01.female'
    return gender_labels

def copy_images_to_gender_folders(base_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    male_folder = os.path.join(output_folder, 'male')
    female_folder = os.path.join(output_folder, 'female')
    
    os.makedirs(male_folder, exist_ok=True)
    os.makedirs(female_folder, exist_ok=True)

    for root, dirs, files in os.walk(base_folder):
        if 'Label.txt' in files:
            label_file = os.path.join(root, 'Label.txt')
            gender_labels = parse_label_file(label_file)
            
            archive_folder = os.path.join(root)
            for img_file in tqdm(glob.glob(os.path.join(archive_folder, '*')),desc=f'{root}'):
                img_name = os.path.basename(img_file)
                parts = img_name.split('_')
                index = parts[0]
                
                if index in gender_labels:
                    if gender_labels[index] == '00.male':
                        shutil.copy(img_file, male_folder)
                    elif gender_labels[index] == '01.female':
                        shutil.copy(img_file, female_folder)

# Example usage
base_folder = '/workspace/PM_source/work/PETA dataset'  # 여러 폴더가 있는 최상위 폴더 경로
output_folder = '/workspace/PM_source/work/PETA_gender_classification_dataset/01.train'  # 결과를 저장할 폴더 경로

copy_images_to_gender_folders(base_folder, output_folder)