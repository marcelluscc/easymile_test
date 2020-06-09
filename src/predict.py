import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.fashion_mnist import load_data
import configparser

(_, _), (test_images, test_labels) = load_data()
test_images = test_images / 255

def predict_model(model):
    result = model.predict_classes(test_images[0:10])
    print("predicted result : ", result)
    print("actual result : ", test_labels[0:10])

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("src/.env")
    
    with tf.compat.v1.Session():
        model = keras.models.load_model(config.get("Inference", "model_filename"))
        predict_model(model)