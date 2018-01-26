import h5py
from keras.preprocessing.image import ImageDataGenerator

bottleneck = False

if bottleneck:
    from keras.models import load_model

    model = load_model("head_model.h5")

else:
    from projects.capstone.resnet50 import ResNet50
    from keras.models import Model
    from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

    print("BUILDING BODY...")
    body = ResNet50(input_shape=(300, 300, 3))
    head = body.output
    head = BatchNormalization(axis=3)(head)
    head = GlobalAveragePooling2D()(head)
    head = Dense(2, activation="softmax")(head)
    model = Model(body.input, head)
    print("LOADING WEIGHTS ...")
    model.load_weights("full_model.h5")

BATCH_SIZE = 8
gen = ImageDataGenerator()
test_batches = gen.flow_from_directory("test", model.input_shape[1:3], batch_size=BATCH_SIZE,
                                       shuffle=False, class_mode=None)


if bottleneck:
    with h5py.File("300_bottlenecks.h5") as hf:
        X_test = hf["test"][:]
    y_test = model.predict(X_test)

else:
    y_test = model.predict_generator(test_batches, test_batches.nb_sample)


import pandas as pd
subm = pd.read_csv("sample_submission.csv")

ids = [int(x.split("\\")[1].split(".")[0]) for x in test_batches.filenames]

for i in range(len(ids)):
    subm.loc[subm.id == ids[i], "label"] = y_test[:,1][i]

subm.to_csv("submission4.csv", index=False)
subm.head()
clipped = y_test.clip(min=0.02, max=0.98)
for i in range(len(ids)):
    subm.loc[subm.id == ids[i], "label"] = clipped[:,1][i]

subm.to_csv("submission4_clipped.csv", index=False)
subm.head()