import tensorflow as tf
import numpy as np
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

TEXT = open('data/train_text/rj.txt').read()
vocab = sorted(set(TEXT))
char2idx = {char: i for i, char in enumerate(vocab)}
idx2char = np.array(vocab)

TOKENIZED = np.array([char2idx[c] for c in TEXT])
TENSOR_DATASET = tf.data.Dataset.from_tensor_slices(TOKENIZED)

seq_length = 100
examples_per_epoch = len(TOKENIZED) // seq_length
SEQUENCES = TENSOR_DATASET.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


DATASET = SEQUENCES.map(split_input_target)

BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE
BUFFER_SIZE = 10000
DATASET = DATASET.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab),
                              output_dim=256,
                              batch_input_shape=[64, None]),
    tf.keras.layers.LSTM(units=1024,
                         return_sequences=True,
                         stateful=True),
    tf.keras.layers.Dense(len(vocab))
])

model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           logits,
                                                           from_logits=True)


model.compile(optimizer='adam', loss=loss)

# checkpoint_dir = 'training_checkpoints/rj'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
#
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
#                                                          save_weights_only=True)

# history = model.fit(DATASET, epochs=100)