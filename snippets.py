import os
import zipfile
from typing import Optional, Tuple, List, Union
import csv
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random
from typing import List

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

	def __init__(self, accuracy: float) -> None:
		self.min_accuracy = accuracy

	def on_epoch_end(self, epoch, logs={}) -> None:
		if logs["acc"] > self.min_accuracy:
			print("Reached {}% accuracy. The training is cancelled".format(self.min_accuracy*100))
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

def pick_files_from_directory(dir: str, k: int = 1) -> List[str]:

	images = random.choices(os.listdir(dir), k=k)
	path_to_images = [os.path.join(dir, image) for image in images]

	return path_to_images

def visualize_convolutions(model: tf.keras.models.Model, image: str) -> None:

	# Let's define a new Model that will take an image as input, and will output
	# intermediate representations for all layers in the previous model after
	# the first.
	successive_outputs = [layer.output for layer in model.layers[1:]]
	visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

	# Let's prepare a random input image from the training set.
	img = load_img(image, target_size=(150, 150))  # this is a PIL image

	x = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
	x = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

	# Rescale by 1/255
	x /= 255.0

	# Let's run our image through our network, thus obtaining all
	# intermediate representations for this image.
	successive_feature_maps = visualization_model.predict(x)

	# These are the names of the layers, so can have them as part of our plot
	layer_names = [layer.name for layer in model.layers]


	# -----------------------------------------------------------------------
	# Now let's display our representations
	# -----------------------------------------------------------------------
	for layer_name, feature_map in zip(layer_names, successive_feature_maps):
		if len(feature_map.shape) == 4:
			#-------------------------------------------
			# Just do this for the conv / maxpool layers, not the fully-connected layers
			#-------------------------------------------
			n_features = feature_map.shape[-1]  # number of features in the feature map
			size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
			# We will tile our images in this matrix
			display_grid = np.zeros((size, size * n_features))
			#-------------------------------------------------
			# Postprocess the feature to be visually palatable
			#-------------------------------------------------
			for i in range(n_features):
				x  = feature_map[0, :, :, i]
				x -= x.mean()
				x /= x.std()
				x *=  64
				x += 128
				x  = np.clip(x, 0, 255).astype('uint8')
				display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

			#-----------------
			# Display the grid
			#-----------------
			scale = 20./n_features
			plt.figure(figsize=(scale*n_features, scale))
			plt.title(layer_name)
			plt.grid(False)
			plt.imshow(display_grid, aspect='auto', cmap='viridis')