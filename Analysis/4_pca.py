import numpy as np
from imgtoarray import load_images, draw_images
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 로드
apple = load_images("Training\Apple Red 1") # 492장
banana = load_images("Training\Banana 1") # 490장
cherry = load_images("Training\Cherry 1") # 492장
fruit = np.concatenate((apple, banana, cherry), axis=0) # 1474장

# 주성분 분석과 주성분 시각화
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruit.reshape(-1, 100*100))
print(pca.components_.shape)
print(pca.explained_variance_ratio_)
draw_images(pca.components_.reshape(-1, 100, 100), ncols=10)