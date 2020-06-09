# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.fashion_mnist import load_data
import numpy as np

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images / 255
test_images = test_images / 255

def create_model(config):
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

def train(model, optimizer, checkpoint, config):
    model.compile(optimizer=optimizer,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

    for epoch in range(config.getint("Training", "epoch")):
        model.fit(train_images, train_labels)
        checkpoint.save(config.get("Training", "checkpoint_prefix"))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nModel accuracy:', test_acc)

if __name__=="__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read("src/.env")
    model = create_model(config)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    train(model, optimizer, checkpoint, config)