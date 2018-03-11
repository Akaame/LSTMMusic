
# datayi oku
import numpy as np
data = np.load("data.npy")
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras_gradient_noise import add_gradient_noise
# model olustur


def get_model(input_shape, output_shape):
    m = Sequential()
    m.add(LSTM(1024, return_sequences=True, input_shape=input_shape))
    m.add(Dropout(0.2))
    m.add(LSTM(1024))
    #m.add(Dense(512))
    m.add(Dense(256,activation="relu"))
    m.add(Dense(output_shape,activation="softmax"))
    return m


print data.shape
data_len = data.shape[0]
feature_len = data.shape[1]  # girdi ve ciktini boyutu
window_size = 100


def data_gen(d, w_size):
    l = d.shape[0]
    for i in range(w_size, l - w_size - 1):
        yield (d[i:i + w_size, :].reshape([1,window_size,-1]), d[i + w_size + 1].reshape([1,-1]))


m = get_model((window_size, feature_len), feature_len)
print m.summary()
from keras.optimizers import RMSprop
rms = add_gradient_noise(RMSprop)
m.compile(optimizer=rms(), loss="categorical_crossentropy",
          metrics=["accuracy"])

m.fit_generator(data_gen(data, window_size), steps_per_epoch=5000, epochs=1)
m.save("model")
# model fit et ve agirliklari sakla
