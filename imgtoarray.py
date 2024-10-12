# 이미지를 3차원 넘파이 배열로 로드할 때 사용할 모듈
import numpy as np
from PIL import Image
import os

def load_images(image_folder):
    
    all_files = os.listdir(image_folder) # 이미지가 저장된 폴더

    image_data = [] # 이미지 리스트 생성

    for file_name in all_files:
        
        if file_name.endswith("_100.jpg"): # 이름이 _100.jpg로 끝나는 파일
            image_path = os.path.join(image_folder, file_name) # 폴더와 이미지 파일 이름을 합쳐서 경로 생성
            img = Image.open(image_path).convert("L") # 이미지를 열고 흑백 처리
            img_array = np.array(img) # 이미지를 넘파이 배열화
            image_data.append(img_array) # 넘파이 배열화된 이미지를 리스트에 추가
    
    image_data = np.stack(image_data, axis=0) # 이미지 리스트를 3차원 넘파이 배열(이미지 개수, 이미지 가로 크기, 이미지 세로 크기)로 변환
    image_data = (255 - image_data) / 255 # 넘파이 배열을 반전하고 정규화
    
    return image_data