import numpy as np
from imgtoarray import load_images, draw_images

# 데이터 로드
apple = load_images("Training\Apple Braeburn 1")
banana = load_images("Training\Banana 1")
cherry = load_images("Training\Cherry 1")
fruit = np.concatenate((apple, banana, cherry), axis=0)

# 각 과일 이미지들의 픽셀 평균 계산
apple_mean = np.mean(apple, axis=0)
banana_mean = np.mean(banana, axis=0)
cherry_mean = np.mean(cherry, axis=0)
fruit_mean = np.mean(fruit, axis=0)

# 각 과일 이미지들의 픽셀 평균 시각화
means = np.array([fruit_mean, apple_mean, banana_mean, cherry_mean])
means_label = ["fruit_mean", "apple_mean", "banana_mean", "cherry_mean"]
draw_images(means, ratio=3, titles=means_label, axis="on", show="off", save=r"Images\2_means")

# 픽셀 평균을 기준으로, 가까운 값부터 골라내기

# 사과
standard = means[1]
differences =np.round(np.mean(np.abs(fruit - standard), axis=(1,2)), 2)
indexes = np.argsort(differences)
draw_images(fruit[indexes], ncols=50, titles=np.sort(differences), show="off", save=r"Images\2_apples sorted by means")

# 바나나
standard = means[2]
differences =np.round(np.mean(np.abs(fruit - standard), axis=(1,2)), 2)
indexes = np.argsort(differences)
draw_images(fruit[indexes], ncols=50, titles=np.sort(differences), show="off", save=r"Images\2_bananas sorted by means")

# 체리
standard = means[3]
differences =np.round(np.mean(np.abs(fruit - standard), axis=(1,2)), 2)
indexes = np.argsort(differences)
draw_images(fruit[indexes], ncols=50, titles=np.sort(differences), show="off", save=r"Images\2_cherries sorted by means")