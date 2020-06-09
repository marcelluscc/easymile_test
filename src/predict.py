import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import configparser
import argparse

def get_sample_image():
    (_, _), (test_images, test_labels) = load_data()
    return test_images[0].reshape(1, 28, 28, 1) / 255.0, test_labels[0]

def load_image(filename):
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img)
    return img.reshape(1, 28, 28, 1) / 255.0

def predict(model, img):
    result = model.predict_classes(img)
    print("predicted result : ", result)

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("src/.env")

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', help="image file that will be predicted. If not provided, it will predict the first test image from dataset")
    args = parser.parse_args()
    
    if args.filename is None:
        image, label = get_sample_image()
        print("actual result : ", label)
    else:
        image = load_image(args.filename)
    
    with tf.compat.v1.Session():
        model = keras.models.load_model(config.get("Inference", "model_filename"))
        predict(model, image)