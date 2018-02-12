import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def addToMap(num, map):
    if num in map:
        map[num] = map[num] + 1
    else:
        map[num] = 1

def showData(map):
    print(len(width_map))
    print(max(width_map))
    bin = np.arange(0, 700, 50)
    plt.hist(width_map)
    plt.title('histograms for width')
    plt.xlabel('width')
    plt.ylabel('count')
    plt.show()

TEST_DIR = '/Users/LukeLin/Downloads/test/'

width_map = []
height_map = []
for file in os.listdir(TEST_DIR):
    with Image.open(TEST_DIR+file) as img:
        width, height = img.size
        width_map.append(width)
        height_map.append(height)

showData(width_map)
showData(height_map)
