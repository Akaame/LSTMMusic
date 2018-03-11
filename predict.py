from keras.models import load_model
from keras import optimizers
from keras_gradient_noise import add_gradient_noise
noisy = add_gradient_noise(optimizers.RMSprop)
from sklearn.preprocessing import OneHotEncoder
from config import window_size, feature_len
import numpy as np

m = load_model("model", custom_objects={"NoisyRMSprop": noisy})
number_of_notes = 50
rand = np.random.randint(0,feature_len,size=[window_size])
ohe = OneHotEncoder(n_values=feature_len,sparse=False)

music = []
music.extend(list(rand))
for i in range(number_of_notes):
    a = np.array(music[i:i+window_size]).reshape([-1,1])
    rand = ohe.fit_transform(a)
    pred = m.predict(rand.reshape([1,window_size,feature_len]))
    music.append(np.argmax(pred))

music = music[window_size:]
with open("classes.txt","r") as f:
    classes = f.readlines()

# one hot decode yap
# sonra label decode yap 
# karsilik gelen note ve chordlardan stream olustur
# stream'i midi dosyasina yaz
# kaydet
labels = []
for idx in music:
    labels.append(classes[idx])

from music21 import stream, note, chord

s = stream.Stream()
for label in labels:
    if "Note" in label:
        print label
        s.append(note.Note(label.split()[1][:-1]))
    else:
        temp = label[:-2].split()[1:]
        print temp
        s.append(chord.Chord(temp))

s.write("mid","results")

