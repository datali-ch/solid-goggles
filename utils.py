import csv
import datetime
import os
import random
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nltk
import numpy as np
import tensorflow as tf

# Get stopwords as gobal variable
try:
	STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except:
	nltk.download('stopwords')
	STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def unpack_file(file: str, folder: str) -> None:
	""" Unpack zip file into the folder.
		Args:
			file:                 	zip file (full path)
			folder:                 destination folder

		Returns:
			None
	"""

	if not os.path.exists(folder):
		os.mkdir(folder)

	with zipfile.ZipFile(file, 'r') as zip_ref:
		zip_ref.extractall(folder)

	print("Files unzipped to /{}".format(folder))


def load_text_data(data_file: str, text_column: int, label_column: Optional[int] = None, remove_stopwords: bool = False) -> Tuple[List[List[str]], List[Union[str, int]]]:
	""" Load text data from csv file.
		Args:
			data_file:                 			zip file (full path)
			text_column:                		index of text column
			label_column (optional): 			index of label column
			remove_stopwords (optional):		True for removing stopwords, False otherwise

		Returns:
			List[List[str]]						text corpus
			List[Union[str, int]]]				labels
	"""

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


def pick_files_from_directory(directory: str, k: int = 1) -> List[str]:
	""" Randomly choose k files from a directory.
		Args:
			directory:                 	directory (full path)
			k (optional):           number of files to choose

		Returns:
			List[str]:				paths to k random files
	"""

	images = random.choices(os.listdir(directory), k=k)
	path_to_images = [os.path.join(directory, image) for image in images]

	return path_to_images


def plot_sample_images(directory: str, subdirectories: List[str], img_per_row: int) -> None:
	""" Plot random images from each subdirectory.
		Args:
			directory:              parent directory (full path)
			subdirectories: 		folders where files are located
			img_per_row:           	number of images per row

		Returns:
			None
	"""

	fig = plt.gcf()
	fig.set_size_inches(img_per_row*4, len(subdirectories)*4)

	for i, category in enumerate(subdirectories):

		curr_path = os.path.join(directory, category)
		path_to_images = pick_files_from_directory(curr_path, k=img_per_row)

		for j, image_path in enumerate(path_to_images):

			sp = plt.subplot(len(subdirectories), img_per_row,
							 i*img_per_row + j + 1)
			sp.axis("Off")

			img = mpimg.imread(image_path)
			plt.imshow(img)

	plt.show()


def visualize_convolutions(model: tf.keras.models.Model, image: str) -> None:
	""" Vizualize image after passing through different convolutional layers.
	Credits to: https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part1.ipynb
		Args:
			model:              	model with convolutional snd pooling layers
			image: 					image to process

		Returns:
			None
	"""

	# Let's define a new Model that will take an image as input, and will output
	# intermediate representations for all layers in the previous model after
	# the first.
	successive_outputs = [layer.output for layer in model.layers[1:]]
	visualization_model = tf.keras.models.Model(
		inputs=model.input, outputs=successive_outputs)

	# Let's prepare a random input image from the training set.
	img = tf.keras.preprocessing.image.load_img(
		image, target_size=(150, 150))  # this is a PIL image

	# Numpy array with shape (150, 150, 3)
	x = tf.keras.preprocessing.image.img_to_array(img)
	# Numpy array with shape (1, 150, 150, 3)
	x = x.reshape((1,) + x.shape)

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
			# -------------------------------------------
			# Just do this for the conv / maxpool layers, not the fully-connected layers
			# -------------------------------------------
			# number of features in the feature map
			n_features = feature_map.shape[-1]
			# feature map shape (1, size, size, n_features)
			size = feature_map.shape[1]
			# We will tile our images in this matrix
			display_grid = np.zeros((size, size * n_features))
			# -------------------------------------------------
			# Postprocess the feature to be visually palatable
			# -------------------------------------------------
			for i in range(n_features):
				x = feature_map[0, :, :, i]
				x -= x.mean()
				x /= x.std()
				x *= 64
				x += 128
				x = np.clip(x, 0, 255).astype('uint8')
				# Tile each filter into a horizontal grid
				display_grid[:, i * size: (i + 1) * size] = x

			# -----------------
			# Display the grid
			# -----------------
			scale = 20./n_features
			plt.figure(figsize=(scale*n_features, scale))
			plt.title(layer_name)
			plt.grid(False)
			plt.imshow(display_grid, aspect='auto', cmap='viridis')

class AccuracyCallback(tf.keras.callbacks.Callback):

	def __init__(self, accuracy: float) -> None:
		self.min_accuracy = accuracy

	def on_epoch_end(self, epoch: int, logs: Dict = {}) -> None:
		""" Stop training if desired accuracy was reached.
			Args:
				epoch:      epoch index
				logs: 		metric results for this training epoch, and for the validation epoch if validation is performed.

			Returns:
				None
		"""

		if logs["acc"] > self.min_accuracy:
			print("Reached {}% accuracy. The training is cancelled".format(
				self.min_accuracy*100))
			self.model.stop_training = True


def plot_training_progress(history: tf.keras.callbacks.History, metric: str, plot_validation: bool = True) -> None:
	""" Plot training progress as per metric.
		Args:
			history:              			history of trained model
			metric: 						metric to plot
			plot_validation (optional):     True for plotting validation metric, False otherwise

		Returns:
			None
	"""

	plt.plot(history.history[metric])

	if plot_validation:
		plt.plot(history.history['val_' + metric])

	plt.xlabel("Epochs")
	plt.ylabel(metric)

	if plot_validation:
		plt.legend([metric, 'val_' + metric])
	plt.show()


def create_n_grams(corpus: List[List[str]], tokenizer: tf.keras.preprocessing.text.Tokenizer, sequence_len: int, max_docs: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
	""" Split the corpus into n-grams. Shorter sequences will be padded.
		Args:
			corpus:					texts used to create n-grams
			tokenizer: 				fitted tokenizer
			sequence_len: 			n in the n-grams
			max_docs(optional):     maximum number of documents to take from corpus

		Returns:
			np.ndarray: 			predictors, sequences of [i:i+n-1]
			np.ndarray				predictands, sequence of [i+n]
	"""

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

	input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(
		input_sequences, maxlen=sequence_len, padding='pre'))
	predictors, predictands = input_sequences[:, :-1], input_sequences[:, -1]

	return predictors, predictands


def load_glove_embeddings(file: str) -> Dict:
	""" Load GloVe word embeddings from a file
		Args:
			file:              file with GloVe embeddings. File format as in https://nlp.stanford.edu/projects/glove/

		Returns:
			word2vec:          dictionnary mapping word to embedding vector
	"""

	FOLDER = "embeddings"
	unpack_file(file, FOLDER)

	word2vec = {}
	f = open(FOLDER + "/glove.6B.100d.txt")
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype="float32")
		word2vec[word] = coefs

	return word2vec


def create_embedding_layer(word2index: dict, word2vec: dict, input_len: int, max_words: int = int(1e6)) -> tf.keras.layers.Embedding:
	""" Set up pretrained Embedding layer
		Credits to: https://keras.io/examples/pretrained_word_embeddings/
		Args:
			word2index:                         dictionnary mapping words to indices
			word2vec:                           dictionnary mapping words to embedding vectors
			input_len: 							length of input sequences
			max_words (optional):				maximum number of words in a dictionnary

		Returns:
			tf.keras.layers.Embedding           layer with pretrained GloVe word embeddings
	"""

	num_words = min(max_words, len(word2index)) + 1
	emb_dim = word2vec["cucumber"].shape[0]
	emb_matrix = np.zeros((num_words, emb_dim))

	for word, index in word2index.items():
		if index > max_words:
			break
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None:
			emb_matrix[index, :] = embedding_vector

	embedding_layer = tf.keras.layers.Embedding(
		num_words, emb_dim, input_length=input_len, trainable=False
	)  # Do not update word embeddings
	embedding_layer.build((None,))
	embedding_layer.set_weights([emb_matrix])

	return embedding_layer


def classify_sentence(model: tf.keras.models.Model, tokenizer: tf.keras.preprocessing.text.Tokenizer, sentence: str, max_sequence_len: int) -> int:
	""" Classify sentence according to the trained model.
		Args:
			model:                  classification model for text data
			tokenizer:              tokenizer for this model
			sentence: 				sentence to classify
			max_sequence_len:		sequence len used to train the model

		Returns:
			int          			predicted class
	"""

	sentence = [[word for word in sentence.split() if word not in STOPWORDS]]
	sequence = tokenizer.texts_to_sequences(sentence)
	padded = tf.keras.preprocessing.sequence.pad_sequences(
		sequence, padding="post", maxlen=max_sequence_len)

	return np.argmax(model.predict(padded))


def generate_text(model: tf.keras.models.Model, tokenizer: tf.keras.preprocessing.text.Tokenizer, seed_text: str, next_words: int) -> str:
	""" Generate the text starting with seed_text.
		Args:
			model:                  multiclass classification model for text data
			tokenizer:              tokenizer for this model
			seed_text: 				starting sentence
			next_words:				numer of words to generate

		Returns:
			str          			generated text
	"""

	index_to_word = {index: word for word,
					 index in tokenizer.word_index.items()}
	sequence_len = model.layers[0].input_length

	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = np.array(tf.keras.preprocessing.sequence.pad_sequences(
		[token_list], maxlen=sequence_len, padding='pre'))

	for i in range(next_words):

		curr_sequence = token_list[:, i:i+sequence_len]
		predicted = model.predict_classes(curr_sequence, verbose=0)
		token_list = np.append(
			token_list, np.reshape(predicted, (1, 1)), axis=1)

	generated_text = [index_to_word[index]
					  for index in token_list[0] if index > 0]

	return " ".join(generated_text)


def find_year_index(dates: List[datetime.date], start_year: int, end_year: Optional[int] = None) -> Tuple[int]:
	""" Find index of a year.
	Important: this funcion will return index for 1st January of given year.
		Args:
			dates:                  list of dates, sorted
			start_year:             start year
			end_year (optional): 	end year

		Returns:
			Tuple[int]          	indices for start and end year
	"""

	start = datetime.date(start_year, 1, 1)
	for start_index, date in enumerate(dates):
		if date == start:
			break

	if end_year:
		end = datetime.date(end_year, 1, 1)
		for end_index, date in enumerate(dates[start_index:]):
			if date == end:
				break

		end_index += start_index
	else:
		end_index = len(dates)

	return start_index, end_index


def preprocess_timeseries(series: List[float], window_size: int, batch_size: int, shuffle_buffer: int) -> tf.data.Dataset:
	""" Process time series for forecasting in RNN.
		Args:
			series:                 observations
			window_size:            number of observations to form sample
			batch_size: 			number of samples to load in batch
			shuffle_buffer: 		number of samples to choose from when shuffling data

		Returns:
			tf.data.Dataset:        windowed, split, and shuffled data for training
	"""

	series = tf.expand_dims(series, axis=-1)
	dataset = tf.data.Dataset.from_tensor_slices(series)
	dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))
	dataset = dataset.shuffle(buffer_size=shuffle_buffer)
	dataset = dataset.map(lambda w: (w[:-1], w[1:]))
	dataset = dataset.batch(batch_size).prefetch(1)

	return dataset


def forecast_timeseries(model: tf.keras.models.Model, series: List[float], window_size: int) -> List[float]:
	""" Forecast time series one step ahead.

		Args:
			model:                 	forecasting model, one step ahead
			series:            		observed variables
			window_size: 			number of observation forming sample, same as for training model

		Returns:
			List[float]:			n+1 forecasts
	"""

	series = np.array(series)[..., np.newaxis]

	dataset = tf.data.Dataset.from_tensor_slices(series)
	dataset = dataset.window(window_size, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda w: w.batch(window_size))
	dataset = dataset.batch(32).prefetch(1)
	forecast = model.predict(dataset)

	return forecast