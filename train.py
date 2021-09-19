import argparse

import mlflow
import mlflow.tensorflow
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from mlflow_callback import MlFlowCallback

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir="/app/data/",
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size")
    parser.add_argument("--epochs")
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    epochs = int(args.epochs)

    with mlflow.start_run():

        mlflow.tensorflow.autolog()
        
        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_test = ds_test.batch(batch_size)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        model = Sequential([
            Flatten(input_shape=(28, 28)), 
            Dense(128, activation="relu"), 
            Dense(10)
        ])

        model.compile(
            optimizer=Adam(0.001),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy()],
        )

        model.fit(
            ds_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=ds_test,
            callbacks=[MlFlowCallback()],
        )
