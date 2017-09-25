import _pickle as pkl

import cv2
import numpy as np

"""

INPUT: two satellita images
OUTPUT: wind speed in kts

Sample Data:
year|basin|storm|time|wind|pressure|file_ir|file_wv|file_cnt|n_images
---|---|---|---|---|---|---|---|---|---
2006|ATL|01L.ALBERTO|2006-06-11 13:00:45|40|1002.0|20060611.1345.goes12.x.ir1km.01LALBERTO.40kts-1002mb-236N-879W.100pc.jpg|20060611.1345.goes12.x.wv1km.01LALBERTO.40kts-1002mb-236N-879W.100pc.jpg|2|1
"""
with open('./data/all_obs.pkl', 'rb') as f:
    df = pkl.load(f)


def read_and_crop(file):
    img = cv2.imread(file)
    if img is None:
        return None
    # original img: 1024 x 1024 > new img: central 800 x 800
    img = img[112:912, 112:912]
    img = cv2.resize(img, (150, 150))
    # normalize all channels
    return img / 255.0


# generator function for batches. randomly loads pairs of images from the full dataset
def gen_train(batch_size=32):
    while True:
        all_img_ir = []
        all_img_wv = []
        all_labels = []
        x = np.random.choice(np.arange(len(df)), batch_size)
        for i in x:
            f = df.iloc[i]
            img_ir = read_and_crop(f['file_ir'])
            img_wv = read_and_crop(f['file_wv'])
            if img_wv is None or img_ir is None:
                continue
            label = f['wind']
            all_img_ir.append(img_ir)
            all_img_wv.append(img_wv)
            all_labels.append(label)

        yield [np.asarray(all_img_ir), np.asarray(all_img_wv)], np.asarray(all_labels)


import keras
from keras.models import Model
from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense, Input

test_X_Y = next(gen_train(32 * 5))


def base_cnn():
    x = Input(shape=(150, 150, 3))
    i = Conv2D(64, (3, 3), activation='relu')(x)
    i = MaxPooling2D()(i)
    i = Conv2D(64, (3, 3), activation='relu')(i)
    i = MaxPooling2D()(i)
    i = Conv2D(128, (3, 3), activation='relu')(i)
    i = MaxPooling2D()(i)
    i = Conv2D(128, (3, 3), activation='relu')(i)
    i = MaxPooling2D()(i)
    i = Conv2D(256, (3, 3), activation='relu')(i)
    i = Dropout(0.5)(i)
    i = Flatten()(i)
    return x, i


model_combined = keras.models.load_model('./data/weights.00-311.79.hdf5')

input_ir, model_ir = base_cnn()
input_water_vapor, model_water_vapor = base_cnn()

architecture_combined = keras.layers.concatenate([model_ir, model_water_vapor])
architecture_combined = Dense(256)(architecture_combined)
architecture_combined = Dense(1)(architecture_combined)

model_combined = Model(inputs=[input_ir, input_water_vapor], outputs=[architecture_combined])
model_combined.compile(loss=keras.losses.mean_squared_error,
                       optimizer=keras.optimizers.Adadelta(),
                       metrics=['mean_squared_error'])

cb = [
    keras.callbacks.EarlyStopping(min_delta=0.5, patience=3, monitor='mean_squared_error'),
    keras.callbacks.ModelCheckpoint(filepath='./data/weights.{epoch:02d}-{mean_squared_error:.2f}.hdf5'),
]
model_combined.fit_generator(gen_train(), steps_per_epoch=100, epochs=100, callbacks=cb, validation_data=test_X_Y)
