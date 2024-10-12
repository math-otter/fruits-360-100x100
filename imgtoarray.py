# 이미지를 3차원 넘파이 배열로 로드할 때 사용할 모듈
import numpy as np
from PIL import Image
import os

def load_images(image_folder):
    
    all_files = os.listdir(image_folder)

    image_data = []

    for file_name in all_files:
        
        if file_name.endswith("_100.jpg"):
            image_path = os.path.join(image_folder, file_name)
            img = Image.open(image_path).convert("L")
            img_array = np.array(img)
            image_data.append(img_array)
    
    image_data = np.stack(image_data, axis=0)
    image_data = (255 - image_data) / 255
    
    return image_data