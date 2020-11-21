import os
import json
import nltk

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics
from tensorflow.keras import callbacks
from tensorflow.python.keras.engine import input_layer

from tqdm import tqdm

checkpoint_path = "checkpoint/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

with open('train_data.json', encoding='utf8') as f:
    train = json.load(f)


with open('test_data.json', encoding='utf8') as f:
    test = json.load(f)


tokens = [t for d in train for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

layer = 10000 # word layers
words = [f[0] for f in text.vocab().most_common(layer)]

def term_frequency(doc, pbar):
    if pbar: pbar.update()
    return [doc.count(word) for word in words]

print('train Data')
pbar = tqdm(total=len(train))
train_x = [term_frequency(d, pbar) for d, _ in train]

print('test Data')
pbar = tqdm(total=len(test))
test_x = [term_frequency(d, pbar) for d, _ in test]
train_y = [c for _, c in train]
test_y = [c for _, c in test]

X = np.asarray(train_x).astype('float32')
Y = np.asarray(train_y).astype('float32')

testX = np.asarray(train_x).astype('float32')
testY = np.asarray(train_y).astype('float32')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_layer=(layer, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizers=optimizers.RMSprop(lr=0.001))

model.fit(X, Y, epochs=10, batch_size=512, callbacks=[cp_callback])

results = model.evaluate(testX, testY)

print(results)
