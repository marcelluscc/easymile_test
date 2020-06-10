# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets.fashion_mnist import load_data
import configparser
import argparse
import matplotlib.pyplot as plt

def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)) / 255
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)) / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels

def create_model(config):
    model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu', kernel_initializer='he_uniform'),
        Dense(10, activation='softmax')
    ])
	# compile model
    opt = keras.optimizers.SGD(lr=config.getfloat("HyperParam", "learning_rate"),
                               momentum=config.getfloat("HyperParam", "momentum"))

    model.compile(optimizer=opt,
                 loss="categorical_crossentropy",
                 metrics=['accuracy', 'Precision'])
    return model

def train(model, config, train_images, train_labels, test_images, test_labels, callbacks):
    history = model.fit(train_images, train_labels, batch_size=32, verbose=1,
                        epochs=config.getint("HyperParam", "epochs"), callbacks=callbacks,
                        validation_data=(test_images, test_labels))
    return history

def show_plot(history, metric_name):
    plt.plot(history.history[metric_name])
    plt.plot(history.history['val_'+metric_name])
    plt.title('model ' + metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def visualize(history):
    show_plot(history, 'acc')
    show_plot(history, 'loss')
    show_plot(history, 'precision')

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("src/.env")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', action='store_true', help="use previous checkpoint if available. Default = False", default=False)
    args = parser.parse_args()
    
    with tf.compat.v1.Session():
        model = create_model(config)

        train_images, train_labels, test_images, test_labels = load_dataset()
    
        if (args.checkpoint):
            if (tf.train.latest_checkpoint(config.get("Training", "checkpoint_dir"))):
                status = model.load_weights(config.get("Training", "checkpoint_filename"))
                print("Checkpoint has been loaded.")
            else:
                print("No checkpoint found.")

        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=config.getint("HyperParam", "early_stopping_patience"))
        mc = ModelCheckpoint(config.get("Training", "checkpoint_filename"), save_weights_only=True)
        callbacks=[es, mc]
        history = train(model, config, train_images, train_labels, test_images, test_labels, callbacks)
        model.save(config.get("Inference", "model_filename"))
        visualize(history)
        