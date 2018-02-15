import os
from PIL import Image
from matplotlib import pyplot as plt


def addToMap(num, map):
    if num in map:
        map[num] = map[num] + 1
    else:
        map[num] = 1

def showData(map, title, xlabel):
    print(len(map))
    print(max(map))
    plt.hist(map)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.show()

TEST_DIR = '/Users/LukeLin/Downloads/test/'

width_map = []
height_map = []
w = 0
h = 0
for file in os.listdir(TEST_DIR):
    with Image.open(TEST_DIR+file) as img:
        width, height = img.size
        width_map.append(width)
        height_map.append(height)
        if height > h:
            h = height
            print('height max', height, file)

        if width > w:
            w = width
            print('width max', width, file)

showData(width_map, 'Test Image Width distribution', 'Width')
showData(height_map, 'Test Image Height distribution', 'Height')
