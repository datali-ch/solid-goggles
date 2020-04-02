import os
import zipfile
from typing import Optional, Tuple, List, Union
import csv
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt

# Get stopwords as gobal variable
try:
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except:
    nltk.download('stopwords')
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def unpack_file(file: str, folder: str) -> None:
    if not os.path.exists(folder):
        os.mkdir(folder)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(folder)

    print("Files unzipped to /{}".format(folder))


def load_text_data(data_file: str, text_column: int, label_column: Optional[int] = None, remove_stopwords: bool=False) -> Tuple[List[List[str]], List[Union[str, int]]]:

    texts = []
    labels = []

    with open(data_file, "r") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        for row in reader:
            sentence = row[text_column].lower().split()
            if remove_stopwords:
                sentence = [word for word in sentence if word not in STOPWORDS]
            texts.append(sentence)

            if label_column is not None:
                labels.append(row[label_column])

    return texts, labels

class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}) -> None:
        if logs["acc"] > 0.9:
            print("Reached 90% accuracy. The training is cancelled")
            self.model.stop_training = True


def plot_graphs(history: tf.keras.callbacks.History, metric: str) -> None:

    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()
