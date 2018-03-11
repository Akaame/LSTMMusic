
# glob ile dosyalari oku

# Iki kotu varsayim var
from glob import glob
import music21 as midi
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pprint import pprint

file_list = glob("data/*")
data_raw = []

for file_name in file_list:
    file_data = midi.converter.parse(file_name) # dosyayi oku
    func = lambda x: isinstance(x, midi.chord.Chord) or isinstance(x, midi.note.Note)
    # nota mi akor mu
    if len(file_data) == 1:
        data_raw.extend(filter(func,file_data.flat.notes))
        # eger sarki cok parcadan olusmuyorsa direk notalari al
    else:
        data_raw.extend(filter(func,file_data.parts[0].recurse()))   
        # eger sarki cok parcadan olusuyorsa ilk parcanin notalarini al

func = lambda x: str(x)
data_raw = map(func, data_raw)
le = LabelEncoder()
data = le.fit_transform(data_raw).reshape([-1,1])
"""
with open("classes.txt","w+") as f:
    f.writelines(map(lambda x: str(x) +"\n", le.classes_))
"""
ohe = OneHotEncoder(sparse=False)
data = ohe.fit_transform(data)
np.save("data", data)
