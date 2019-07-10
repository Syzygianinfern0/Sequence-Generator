import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, Dense, Dropout, Bidirectional
from tensorflow.python.keras.optimizers import Adam
import numpy as np

with open('data/original text/shakespeare-romeo-juliet.txt', 'r') as handler:
    # Data from http://www.ebooksread.com/dl2.php?action=output_file&id=1513&ext=txt&f=2ws1610&a_id=2397
    TEXT = handler.read()

vocab = sorted(set(TEXT))

tokenizer = {}
reverse = np.array(vocab)

for char, i in zip(vocab, range(len(vocab))):
    tokenizer[char] = i
