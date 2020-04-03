import os
import zipfile
from typing import Optional, Tuple, List, Union
import csv
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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


def plot_graphs(history: tf.keras.callbacks.History, metric: str, plot_validation: bool = True) -> None:

    plt.plot(history.history[metric])

    if plot_validation:
        plt.plot(history.history['val_' + metric])

    plt.xlabel("Epochs")
    plt.ylabel(metric)

    if plot_validation:
        plt.legend([metric, 'val_' + metric])
    plt.show()


def create_n_grams(corpus: List[List[str]], tokenizer: tf.keras.preprocessing.text.Tokenizer, sequence_len: int, max_docs: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:

    input_sequences = []

    for j, phrase in enumerate(corpus):
        token_list = tokenizer.texts_to_sequences([phrase])[0]
        if len(token_list) < 2:
            continue
        elif len(token_list) < sequence_len:
            input_sequences.append(token_list)

        for i in range(1, len(token_list) - sequence_len):
            n_gram_sequence = token_list[i:i + sequence_len]
            input_sequences.append(n_gram_sequence)

        if max_docs and j > max_docs:
            break

    input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=sequence_len, padding='pre'))
    predictors, predictands = input_sequences[:,:-1], input_sequences[:,-1]

    return predictors, predictands


def generate_text(model: tf.keras.models.Model, tokenizer: tf.keras.preprocessing.text.Tokenizer, seed_text: str, next_words: int) -> str:

    index_to_word = {index:word for word,index in tokenizer.word_index.items()}
    sequence_len = model.layers[0].input_length

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = np.array(tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=sequence_len, padding='pre'))

    for i in range(next_words):

        curr_sequence = token_list[:,i:i+sequence_len]
        predicted = model.predict_classes(curr_sequence, verbose=0)
        token_list = np.append(token_list, np.reshape(predicted, (1,1)), axis=1)

    generated_text = [index_to_word[index] for index in token_list[0] if index > 0]

    return " ".join(generated_text)