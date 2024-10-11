import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

image_folder = "Training\Banana 1"
all_files = os.listdir(image_folder)
print(all_files[0])


# image_files = [f"r0_{i}_100.jpg" for i in range(0, 100)]

# image_data = []
# for image_file in image_files:
#     image_path = os.path.join(image_folder, image_file)
#     try:
#         img = Image.open(image_path).convert("L")
#         img_array = np.array(img)
#         image_data.append(img_array)
#     except FileNotFoundError:
#         continue
# image_data = np.stack(image_data, axis=0)

# plt.imshow(image_data[5])
# plt.show()