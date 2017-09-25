import _pickle as pkl

import cv2
import numpy as np

with open('./data/all_obs.pkl', 'rb') as f:
    df = pkl.load(f)


def r(file):
    img = cv2.imread(file)
    if img is None:
        return None
    # 1024 x 1024 > 800 x 800
    # -224
    img = img[112:912, 112:912]
    img = cv2.resize(img, (150, 150))
    return img / 255.0


batch_size = 32


def gen_train():
    while True:
        all_img_ir = []
        all_img_wv = []
        all_labels = []
        x = np.random.choice(np.arange(len(df)), batch_size)
        for i in x:
            f = df.iloc[i]
            img_ir = r(f['file_ir'])
            img_wv = r(f['file_wv'])
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


def base_nn():
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


i_ir, m_ir = base_nn()
i_wv, m_wv = base_nn()

m_con = keras.layers.concatenate([m_ir, m_wv])
m_con = Dense(256)(m_con)
m_con = Dense(1)(m_con)

m = Model(inputs=[i_ir, i_wv], outputs=[m_con])

m.summary()

m.compile(loss=keras.losses.mean_squared_error,
          optimizer=keras.optimizers.Adadelta(),
          metrics=['mean_squared_error'])

cb = [
    keras.callbacks.EarlyStopping(min_delta=0.5, patience=3, monitor='mean_squared_error'),
    keras.callbacks.ModelCheckpoint(filepath='./data/weights.{epoch:02d}-{mean_squared_error:.2f}.hdf5'),
]
m.fit_generator(gen_train(), steps_per_epoch=100, epochs=100, callbacks=cb)
