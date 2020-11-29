from PIL import Image

img = Image.open('D:/code/study/image/lena.jpg')

img.show()

print('图片的格式',img.format)
print('图片的大小',img.size)
print(img.height, img.width)
print(img.getpixel(100,100))
print(img)