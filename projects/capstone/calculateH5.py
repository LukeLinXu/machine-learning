from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py


def create_h5(MODEL, image_size, lambda_func=None):
    print(MODEL, 'start')
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("input/train", image_size, shuffle=False,
                                              batch_size=16)
    test_generator = gen.flow_from_directory("input/test1", image_size, shuffle=False,
                                             batch_size=16, class_mode=None)

    train = model.predict_generator(train_generator, train_generator.samples/16)
    test = model.predict_generator(test_generator, test_generator.samples/16)
    with h5py.File("feature_%s.h5"%MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


create_h5(ResNet50, (224, 224))
create_h5(Xception, (299, 299), xception.preprocess_input)
create_h5(InceptionV3, (299, 299), inception_v3.preprocess_input)
create_h5(VGG16, (224, 224))
create_h5(VGG19, (224, 224))