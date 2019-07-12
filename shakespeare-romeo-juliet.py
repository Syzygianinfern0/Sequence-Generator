import tensorflow as tf
from tensorflow._api.v2.v2 import keras
import numpy as np

with open('data/original text/shakespeare-romeo-juliet.txt', 'r') as handler:
    # Data from http://www.ebooksread.com/dl2.php?action=output_file&id=1513&ext=txt&f=2ws1610&a_id=2397
    TEXT = handler.read()

"""
Manual Tokenization
~~~~~~ ~~~~~~~~~~~~
vocab = sorted(set(TEXT))
tokenizer = {}
reverse = np.array(vocab)

for char, i in zip(vocab, range(len(vocab))):
    tokenizer[char] = i"""

# Tokenization
# ~~~~~~~~~~~~
vocab = sorted(set(TEXT))
tokenizer = keras.preprocessing.text.Tokenizer(filters=None,  # Allow all punctuation and special characters
                                               lower=False,  # <--- Preserve names and stuff
                                               split='',  # <--- To look at characters
                                               char_level=True,  # <--- To work on a character level of split
                                               oov_token='<OOV>')

tokenizer.fit_on_texts(TEXT)

CVT_TEXT = tokenizer.texts_to_sequences(TEXT)

input_length = 100
no_batches = len(CVT_TEXT) // (input_length + 1)

# SEQUENCES
# ~~~~~~~~~
SEQUENCES = []

for i in np.arange(start=0,
                   stop=len(CVT_TEXT),
                   step=input_length + 1):
    SEQUENCES.append(CVT_TEXT[i:i + input_length + 1])

"""
# To show the sequences
for line in SEQUENCES[:5]:
    print(f"{repr(''.join(tokenizer.sequences_to_texts(line)))}")"""

# DATA CREATION
# ~~~~ ~~~~~~~~
INPUT = []
OUTPUT = []
for chunk in SEQUENCES:
    INPUT.append(chunk[:-1])
    OUTPUT.append(chunk[1:])

"""
# Show the 1st entry
print(f"Input  : {repr(''.join(tokenizer.sequences_to_texts(INPUT[0])))}")
print(f"Output : {repr(''.join(tokenizer.sequences_to_texts(OUTPUT[0])))}")"""

model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=len(vocab),
                           output_dim=256,
                           batch_input_shape=[1, None]
                           ),
    keras.layers.Bidirectional(keras.layers.LSTM(units=512,
                                                 return_sequences=True,
                                                 stateful=True,
                                                 recurrent_initializer='glorot_uniform')),
    keras.layers.Dropout(0.2),
    keras.layers.Bidirectional(keras.layers.LSTM(units=512,
                                                 return_sequences=True,
                                                 stateful=True,
                                                 recurrent_initializer='glorot_uniform')),
    keras.layers.Dense(units=len(vocab) / 2,
                       activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.2)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(len(vocab))

])

print(model.summary())


# predicted = model.predict(INPUT[:1])


def prediction(predicted):
    string = ''
    for i in range(len(predicted)):
        string += tokenizer.sequences_to_texts(tf.random.categorical(predicted[i], 1).numpy())[0]
    return string


# def loss(labels, logits):
#     return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy)

model.fit(INPUT, OUTPUT)
