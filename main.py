
####################################################################################################
# https://www.tensorflow.org/hub - TensorFlow Hub is a repository of trained machine learning models
# With transfer learning we reuse parts of an already trained model and change the final layer,
# or several layers, of the model, and then retrain those layers on our own dataset.
####################################################################################################

import tensorflow as tf
import matplotlib.pylab as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from keras import layers
import logging
import numpy as np
import PIL.Image as Image





logger = tf.get_logger()
logger.setLevel(logging.ERROR)






if __name__ == '__main__':

# MobileNet is expecting images of 224 X 224 pixels, in 3 color channels (RGB).
    CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
    IMAGE_RES = 224

    model = tf.keras.Sequential([
        hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    ])

    car_image = Image.open('car.jpg').resize((IMAGE_RES, IMAGE_RES))
    military_form = Image.open('grace.jpg').resize((IMAGE_RES, IMAGE_RES))
    inJava = Image.open('inJava.jpg').resize((IMAGE_RES, IMAGE_RES))

    list_image = [car_image, military_form, inJava]
    normal_list_image = []

    for index, image in enumerate(list_image):
        image = np.array(image) / 255.0
        normal_list_image.append(image)
        print(str(index) + ' ' + str(image.shape))

# ============================= ## Decode the predictions ========================>

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

# models always want a batch of images to process. So here, we add a batch dimension,
# and pass the image to the model for prediction
    for img in normal_list_image:
        result = model.predict(img[np.newaxis, ...])
        print(result.shape)

        predicted_class = np.argmax(result[0], axis=-1)
        print(predicted_class)

        plt.imshow(img)
        plt.axis('off')
        predicted_class_name = imagenet_labels[predicted_class]
        _ = plt.title("Prediction: " + predicted_class_name.title())
        plt.show()










