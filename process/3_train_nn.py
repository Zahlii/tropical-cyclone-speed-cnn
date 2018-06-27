import cv2
import glob
import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Input, AlphaDropout, Dropout
from keras.models import Model

from settings import IMAGE_SIZE

df = pd.read_pickle('../data/all_obs.pkl', compression='gzip')
df['key'] = df.year + '-' + df.basin + '-' + df.storm


def split_df(train=0.7):
    # np.random.seed(1234)
    key_count = df.groupby('key', as_index=False).agg({'basin': 'count'}).sample(frac=1).reset_index(drop=True)
    key_count['cum_sum'] = np.cumsum(key_count.basin)
    total = key_count.cum_sum.iloc[-1]
    total_train = train * total
    first_bigger = (key_count.cum_sum >= total_train).idxmax()

    mask_train = key_count.key[0:first_bigger + 1]
    mask_test = key_count.key[first_bigger + 1:]

    print('Storms Train', len(mask_train))
    print('Storms Test', len(mask_test))

    df_train = df[df.key.isin(mask_train)]
    df_test = df[df.key.isin(mask_test)]

    print('Total Train', len(df_train))
    print('Total Test', len(df_test))

    return df_train, df_test


df_train, df_test = split_df(0.95)


def read(file):
    img = cv2.imread(file)
    if img is None:
        return None
    return img.astype('float32') / 255.0


class Encoder:
    """
    Given a large-scale regression problem with a neural network it can be helpful to instead use output bins.
    E.g. when predicting the age of a person from an image, possible output bins might be np.arange(0,101,1).

    This encoder transforms numerical values (i.e. the age) into a normal probability distribution over these bins
    which then can be used as a target for multi-label classification.

    This means that networks which use this encoder should use "binary_crossentropy" loss together with "sigmoid"
    activation in the last layer.

    Example:

        enc = RegressionToClassificationEncoder(classes=np.arange(0,101,1))
        y = [[35],[28],[16]]
        y_transformed = enc.transform(y) # gives a shape (3 x 100) array

        model = keras.models.Sequential()
        ...
        model.add(enc.get_last_layer()) # Dense(100, activation='sigmoid')

        model.compile(loss=keras.losses.binary_crossentropy, optimizer='Adam',
            metrics=[enc.mean_absolute_error, enc.mean_squared_error])

        model.fit(x_train, y_transformed)

        y_test_transformed = model.predict(x_test)
        y_test = enc.inverse_transform(y_test_transformed)


    """

    def __init__(self, classes):
        self.classes = classes
        self.n_classes = len(self.classes)
        self.std = 3  # np.std(self.classes)
        self.mean = np.mean(self.classes)
        self._class_tensor = K.constant(value=self.classes.reshape(-1, 1), dtype='float32')
        print(self.classes)

    def transform(self, vals):
        vals = np.asarray(vals, dtype='float32')
        n_vals = vals.shape[0]
        e = np.zeros((n_vals, self.n_classes))

        c2 = 2 * self.std * self.std
        # c = 1.0 / np.sqrt(np.pi * c2)

        for i, val in enumerate(vals):
            r = np.exp(-1 * np.square(val - self.classes) / c2)
            # r[r < K.epsilon()] = 0
            e[i, :] = r
        return e

    def inverse_transform(self, vals):
        return (vals / np.sum(vals, axis=1, keepdims=True)).dot(self.classes)

    def _inv_tensor(self, y):
        # y (n_images x 20)
        # sum (n_images x 1)
        # div (n_images x 20)
        # dot (n_images x 20) x (20 x 1) -> (n_images x 1)
        d = (y / K.sum(y, axis=1, keepdims=True))
        z = K.dot(d, self._class_tensor)
        e = K.reshape(z, (-1,))
        return e

    def mean_squared_error(self, y_true, y_pred):
        return keras.losses.mean_squared_error(self._inv_tensor(y_true), self._inv_tensor(y_pred))

    def mean_absolute_error(self, y_true, y_pred):
        return keras.losses.mean_absolute_error(self._inv_tensor(y_true), self._inv_tensor(y_pred))

    def get_last_layer(self):
        return keras.layers.Dense(len(self.classes), activation='sigmoid')


ENC = Encoder(classes=np.arange(0, 201, 5))


# generator function for batches. randomly loads pairs of images from the full dataset
def gen(my_df, batch_size=128, which='both'):
    while True:
        all_img_ir = []
        all_img_wv = []
        all_labels = []

        x = np.random.choice(np.arange(len(my_df)), batch_size)
        for i in x:
            f = my_df.iloc[i]
            label = f['wind']
            if which == 'both':
                img_ir = read(f['file_ir'])
                img_wv = read(f['file_wv'])
                if img_wv is None or img_ir is None:
                    continue
                all_img_ir.append(img_ir)
                all_img_wv.append(img_wv)
                all_labels.append(label)
            elif which == 'ir':
                img_ir = read(f['file_ir'])
                if img_ir is None:
                    continue
                all_img_ir.append(img_ir)
                all_labels.append(label)
            elif which == 'wv':
                img_wv = read(f['file_wv'])
                if img_wv is None:
                    continue
                all_img_wv.append(img_wv)
                all_labels.append(label)

        IR, WV, Y = np.asarray(all_img_ir, dtype='float32'), np.asarray(all_img_wv, dtype='float32'), ENC.transform(
            all_labels)
        if which == 'both':
            yield [IR, WV], Y
        elif which == 'ir':
            yield [IR], Y
        elif which == 'wv':
            yield [WV], Y


def base_cnn2():
    m = keras.applications.InceptionV3(include_top=False, weights=None, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                       pooling='max')
    return m.inputs[0], m.outputs[0]


def base_cnn():
    x = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    i = Conv2D(64, (3, 3), activation='selu', kernel_initializer='lecun_normal', name='conv1')(x)
    i = MaxPooling2D()(i)
    i = Conv2D(64, (3, 3), activation='selu', kernel_initializer='lecun_normal', name='conv2')(i)
    i = MaxPooling2D()(i)
    i = Conv2D(128, (3, 3), activation='selu', kernel_initializer='lecun_normal', name='conv3')(i)
    i = MaxPooling2D()(i)
    i = Conv2D(128, (3, 3), activation='selu', kernel_initializer='lecun_normal', name='conv4')(i)
    i = MaxPooling2D()(i)
    i = Conv2D(256, (3, 3), activation='selu', kernel_initializer='lecun_normal', name='conv5')(i)
    i = MaxPooling2D()(i)
    i = Conv2D(256, (3, 3), activation='selu', kernel_initializer='lecun_normal', name='conv6')(i)
    # i = Conv2D(512, (3, 3), activation='selu', kernel_initializer='lecun_normal')(i)
    i = AlphaDropout(0.3)(i)
    i = Flatten()(i)
    return x, i


if True:
    for m in [ 'wv']:

        gen_train = gen(df_train, which=m)
        gen_test = gen(df_test, which=m)

        # architecture_combined = keras.layers.concatenate([model_ir, model_water_vapor])

        inp, model = base_cnn2()
        model = Dropout(0.4)(model)
        architecture_combined = Dense(256, activation='relu')(model)
        architecture_combined = Dense(ENC.n_classes, activation='sigmoid')(architecture_combined)

        model_combined = Model(inputs=[inp], outputs=[architecture_combined])

        file = glob.glob('../data/' + m + '*h5')
        if len(file) > 0:
            print('Loading %s' % file[0])
            model_combined.load_weights(file[0], by_name=True)

        for l in model_combined.layers:
            if m not in l.name:
                l.name = m + '_' + l.name

        model_combined.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.SGD(momentum=0.9, decay=1e-6),
                               metrics=[ENC.mean_squared_error, ENC.mean_absolute_error])

        print('Parameters', model_combined.count_params())

        cb = [
            # keras.callbacks.EarlyStopping(min_delta=0.5, patience=3, monitor='mean_squared_error'),
            keras.callbacks.ModelCheckpoint(save_weights_only=True, save_best_only=True,
                                            filepath='../data/' + m + '_EPOCH={epoch:02d}_MAE={val_mean_absolute_error:.2f}.h5',
                                            monitor='val_mean_absolute_error', mode='min'),
        ]

        model_combined.fit_generator(gen_train, steps_per_epoch=15, epochs=3, callbacks=cb, validation_data=gen_test,
                                     validation_steps=5)
else:
    gen_train = gen(df_train, which='both')
    gen_test = gen(df_test, which='both')

    file = glob.glob('../data/ir*hdf5')
    model_ir = keras.models.load_model(file[0], compile=False)

    model_ir.save_weights('weights_ir')

    for l in model_ir.layers:
        l.name += 'IR'
    inp_ir = model_ir.inputs[0]
    out_ir = model_ir.layers[-2].output

    file = glob.glob('../data/wv*hdf5')
    model_wv = keras.models.load_model(file[0], compile=False)
    for l in model_wv.layers:
        l.name += 'WV'
    inp_wv = model_wv.inputs[0]
    out_wv = model_wv.layers[-2].output

    architecture_combined = keras.layers.concatenate([out_ir, out_wv])
    architecture_combined = Dense(ENC.n_classes, activation='sigmoid')(architecture_combined)
    model_combined = Model(inputs=[inp_ir, inp_wv], outputs=[architecture_combined])

    model_combined.compile(loss=keras.losses.binary_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=[ENC.mean_squared_error])

    model_combined.summary()

    cb = [
        # keras.callbacks.EarlyStopping(min_delta=0.5, patience=3, monitor='mean_squared_error'),
        keras.callbacks.ModelCheckpoint(
            filepath='../data/combined_weights.{epoch:02d}-{val_mean_squared_error:.2f}.hdf5'),
    ]

    model_combined.fit_generator(gen_train, steps_per_epoch=40, epochs=5, callbacks=cb, validation_data=gen_test,
                                 validation_steps=5)
