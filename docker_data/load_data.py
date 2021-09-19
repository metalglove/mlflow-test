import tensorflow_datasets as tfds

tfds.load('mnist', split=['train','test'], data_dir="/app/data/")