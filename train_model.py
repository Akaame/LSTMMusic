
# datayi oku
import numpy as np
data = np.load("data.npy")
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras_gradient_noise import add_gradient_noise
from config import window_size, k
# model olustur


def get_model(input_shape, output_shape):
    m = Sequential()
    m.add(LSTM(1024, return_sequences=True, input_shape=input_shape))
    #m.add(Dropout(0.2))
    m.add(LSTM(512))
    #m.add(Dropout(0.2))
    m.add(Dense(512))
    m.add(Dense(256,activation="relu"))
    m.add(Dense(output_shape,activation="softmax"))
    return m


print data.shape
data_len = data.shape[0]
feature_len = data.shape[1]  # girdi ve ciktini boyutu


def data_gen(d, w_size, k):
    l = d.shape[0]
    for i in range(w_size, l - w_size- k - 1):
        ret_X = []
        ret_y = []
        for idx in range(k):
            ret_X.append(d[i+idx:i+idx + w_size, :])
            ret_y.append(d[i+idx + w_size + 1])
        yield (np.array(ret_X), np.array(ret_y))

m = get_model((window_size, feature_len), feature_len)
print m.summary()
from keras.optimizers import RMSprop
rms = add_gradient_noise(RMSprop)
m.compile(optimizer=rms(), loss="categorical_crossentropy",
          metrics=["accuracy"])

m.fit_generator(data_gen(data, window_size,k), steps_per_epoch=90, epochs=50)
m.save("model")
# model fit et ve agirliklari sakla
