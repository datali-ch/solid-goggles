import tensorflow as tf
import zipfile
import os
import matplotlib.pyplot as plt

PATH_TO_ZIPFILE = "intel-image-classification.zip"
IMG_DIR = "images"

if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)

    with zipfile.ZipFile(PATH_TO_ZIPFILE, 'r') as zip_ref:
        zip_ref.extractall(IMG_DIR)
    print("Files unzipped to {}".format(IMG_DIR))


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

TRAIN_DIR = "images/seg_train"
TEST_DIR = "images/seg_test"

train_generator = image_generator.flow_from_directory(TRAIN_DIR, target_size=(150, 150), class_mode="categorical", batch_size=32)
test_generator = image_generator.flow_from_directory(TEST_DIR, target_size=(150, 150), class_mode="categorical", batch_size=32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(6, activation="softmax")
])

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs["acc"] > 0.9:
            print("Reached 90% accuracy. The training is cancelled")
            self.model.stop_training = True

my_callback = MyCallback()
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_generator, validation_data=test_generator, epochs=2, verbose=1, callbacks=[my_callback])

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Training and validation accuracy")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training and validation loss")
plt.show()

# To Do
# Show sample images
# Visualize convolutions
# Transfer learning