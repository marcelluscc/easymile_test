# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.fashion_mnist import load_data
import numpy as np
import configparser

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images / 255
test_images = test_images / 255

def create_model(config):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer="adam",
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    return model

def train(model, config):
    for epoch in range(config.getint("Training", "epoch")):
        model.fit(train_images, train_labels)
        model.save_weights(config.get("Training", "checkpoint_prefix"))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nModel accuracy:', test_acc)

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("src/.env")
    
    with tf.Session():
        model = create_model(config)
    
        if (True):
            status = model.load_weights(config.get("Training", "checkpoint_prefix"))

        train(model, config)