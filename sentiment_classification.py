import tensorflow as tf
import zipfile
import os
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

PATH_TO_ZIPFILE = "tweet-sentiment-extraction.zip"
TXT_DIR = "tweets"

if not os.path.exists(TXT_DIR):
    os.mkdir(TXT_DIR)

    with zipfile.ZipFile(PATH_TO_ZIPFILE, 'r') as zip_ref:
        zip_ref.extractall(TXT_DIR)
    print("Files unzipped to {}".format(TXT_DIR))

TRAIN_FILE = "tweets/train.csv"
TEST_FILE = "tweets/test.csv"

texts = []
selected_texts = []
labels =[]
with open(TRAIN_FILE, "r") as file:
    reader = csv.reader(file, delimiter=",")
    next(reader)
    for row in reader:
        texts.append(row[1].split())
        selected_texts.append(row[2].split())
        labels.append(row[3])

# Add stopwords removal
sentiment_map = {
    "positive": 2,
    "neutral": 1,
    "negative": 0,
}

sentiment = [sentiment_map[label] for label in labels]


VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LEN = 50

tokenizer = Tokenizer(oov_token="<oov>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding = "post", maxlen = MAX_LEN)

print(len(padded))
print(len(labels))

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, EMBEDDING_DIM, input_length = MAX_LEN),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation = "relu"),
    tf.keras.layers.Dense(3, activation = "softmax")
])

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs["acc"] > 0.4:
            print("Reached 50% accuracy. The training is cancelled")
            self.model.stop_training = True

my_callback = MyCallback()
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(padded, np.array(sentiment, dtype=int), epochs=2, verbose=1, callbacks=[my_callback])

"""
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Training and validation accuracy")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training and validation loss")
plt.show()

# To Do
# Embeddings (?)
# Transfer learning (?)
"""