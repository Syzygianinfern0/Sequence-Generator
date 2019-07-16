import tensorflow as tf
import numpy as np
import random
import string


class TextSampler:
    def __init__(self,
                 file='data/train_text/rj.txt'):
        """
        Init required Variables
        :param file: Text file to sample from
        """
        # Reading the file
        self._read_file(file)

    def make_model(self,
                   trained_weights='models/rj.h5',
                   embedding_size=256,
                   lstm_units=1024,
                   lstm_layers=1):
        """
        Creates the model that accepts the weights that are trained
        :param trained_weights: Trained weights of file type '.h5'
        :param embedding_size: Dimensionality of the Embedding
        :param lstm_units: Number of units
        :param lstm_layers: Number of stacked layers
        :return: Model with trained weights
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=len(self.vocab),
                                            output_dim=embedding_size,
                                            batch_input_shape=[1, None]))
        for i in range(lstm_layers):
            model.add(tf.keras.layers.LSTM(units=lstm_units,
                                           return_sequences=True,
                                           stateful=True))
        model.add(tf.keras.layers.Dense(len(self.vocab)))
        model.load_weights(trained_weights)
        model.build(tf.TensorShape([1, None]))
        return model

    def generate_text(self, model, temp=1.0):
        """
        Generates text using the model with the passed seed
        :param model: Trained Model
        :param temp: Temperature of predictions (Higher equals more surprising)
        :return: None
        """
        num = int(input("\n\nEnter number of characters to sample (Def:1000) : ") or 1000)
        seed = input("Enter start string (Def: Random letter) : ") or random.choice(string.ascii_uppercase)

        # noinspection PyUnusedLocal
        input_idx = [self.char2idx[c] for c in seed]
        input_idx = tf.expand_dims(input_idx, axis=0)
        gen = []
        model.reset_states()

        for i in range(num):
            pred = model.predict(input_idx)
            pred = tf.squeeze(pred, axis=0)
            pred /= temp
            pred_idx = tf.random.categorical(logits=pred,
                                             num_samples=1)[-1, 0].numpy()
            gen.append(self.idx2char[pred_idx])
            input_idx = tf.expand_dims([pred_idx], axis=0)
        print(f"\n\n\n{seed}{''.join(gen)}")

    def _read_file(self, file):
        """
        Reads the file passed
        :param file: File name as relative path to parent directory
        :return: None
        """
        self.original_text = open(file, 'rb').read().decode('utf-8')
        self.vocab = sorted(set(self.original_text))
        self.char2idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
