import os
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(fill_mode='constant', zoom_range=[4, 4])
#
TEST_FILE = '/Users/LukeLin/Downloads/train/cat.257.jpg'
img = load_img(TEST_FILE)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

shutil.rmtree('preview')
os.makedirs('preview')
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 6:
        break  # otherwise the generator would loop indefinitely

fig = plt.figure()
index = 1


def process(filename, index):
    image = mpimg.imread("preview"+os.sep+filename)
    sub = fig.add_subplot(2, 4, index + 1)
    sub.imshow(image)

image = mpimg.imread(TEST_FILE)
sub = fig.add_subplot(2, 4, 1)
sub.imshow(image)

for file in os.listdir("preview"):
    process(file, index)
    index = index+1

plt.show()

