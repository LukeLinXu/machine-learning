# from resnet50 import ResNet50
from projects.capstone.resnet50 import ResNet50

weights_path = 'resnet50_body.h5'

body = ResNet50(input_shape=(300,300,3), weights_path=weights_path)

for layer in body.layers:
    layer.trainable = False

from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

head = body.output
head = BatchNormalization(axis=3)(head)
head = GlobalAveragePooling2D()(head)
head = Dense(2, activation="softmax")(head)


model = Model(body.input, head)


aug_gen = ImageDataGenerator(horizontal_flip=True,
                             zoom_range=0.05,
                             fill_mode="constant",
                             channel_shift_range=10,
                             rotation_range=5,
                             width_shift_range=0.05,
                             height_shift_range=0.05)

train_batches = aug_gen.flow_from_directory("../../../input/train", model.input_shape[1:3],
                                                      shuffle=True, batch_size=8)
val_batches = aug_gen.flow_from_directory("../../../input/valid", model.input_shape[1:3],
                                                      shuffle=True, batch_size=8)


cb = [ModelCheckpoint("full_model.h5", save_best_only=True, save_weights_only=True)]


model.compile(Adam(), "categorical_crossentropy")
nb_epoch = 3

model.fit_generator(train_batches, train_batches.samples,
                    nb_epoch=nb_epoch,
                    callbacks=cb,
                    validation_data=val_batches, nb_val_samples=val_batches.samples
                   )