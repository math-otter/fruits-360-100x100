import matplotlib.pyplot as plt
from imgtoarray import load_images

bananas = load_images("Training\Banana 1")
apples = load_images("Training\Apple Red 1")

print(bananas.shape)

plt.imshow(apples[0], cmap="gray_r")
plt.show()