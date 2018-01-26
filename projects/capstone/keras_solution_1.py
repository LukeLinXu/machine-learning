import os


os.makedirs('../../../input/train/dog', exist_ok=True)
os.makedirs('../../../input/train/cat', exist_ok=True)

for dir, subdir, files in os.walk("../../../input/train"):
    if len(subdir) == 0:
        continue
    for file in files:
        category = file.split(".")[0]
        os.rename("{}/{}".format(dir,file), "{}/{}/{}".format(dir, category, file))

os.makedirs('../../../input/valid/dog', exist_ok=True)
os.makedirs('../../../input/valid/cat', exist_ok=True)
dogs = [x for x in os.listdir("../../../input/train/dog")]
cats = [x for x in os.listdir("../../../input/train/cat")]
import random
if len(os.listdir("../../../input/valid/dog")) < 1:
    for n in random.sample(range(len(dogs)), 1000):
        os.rename("../../../input/train/dog/{}".format(dogs[n]), "../../../input/valid/dog/{}".format(dogs[n]))

if len(os.listdir("../../../input/valid/cat")) < 1:
    for n in random.sample(range(len(cats)), 1000):
        os.rename("../../../input/train/cat/{}".format(cats[n]), "../../../input/valid/cat/{}".format(cats[n]))