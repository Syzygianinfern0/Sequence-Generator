import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, Dense, Dropout, Bidirectional
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
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
tokenizer = Tokenizer(filters=None,  # Allow all punctuation and special characters
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

model = Sequential([
    Embedding(input_dim=len(vocab),
              output_dim=256),
    Bidirectional(CuDNNLSTM(units=512,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform')),
    Dropout(0.2),
    CuDNNLSTM(units=256),
    Dense(units=len(vocab)/2,
          )

])
