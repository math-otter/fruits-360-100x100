from Analysis.imgtoarray import load_images, draw_images

bananas = load_images("Training\Banana 1")
apples = load_images("Training\Apple Red 1")

print(bananas.shape)

draw_images(bananas[0:8], ncols=4, ratio=3, titles=[n for n in range(8)], axis="on", cmap="gray")