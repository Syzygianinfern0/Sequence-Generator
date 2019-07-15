import tensorflow as tf
import numpy as np


class TextTrainer:
    def __init__(self,
                 file='data/train_text/rj.txt',
                 sequence_length=100,
                 batch_size=64):
        """
        Init all class variables and show some stats
        :param file: File name as relative path to parent directory
        :param sequence_length: Length of train data and labels
        :param batch_size: Batch size to train in
        """
        self.bs = batch_size
        self.seq_len = sequence_length
        # Reading the file
        self._read_file(file)

        # Tokenizing
        tensor_dataset = self._tokenize()

        # Dataset Pipe Lining
        sequences = self._create_dataset(tensor_dataset)

        # Forming Train Data, Labels
        self._create_train_dataset(sequences)

        print(f"\n\nLength of text : {len(self.original_text)} characters \nNumber of unique characters : "
              f"{len(self.vocab)}\n\n")
        print(30 * '-')
        print(f"\n\nFirst few characters : \n{self.original_text[:250]}")

    def make_model(self,
                   embedding_size=256,
                   lstm_units=1024,
                   lstm_layers=1,
                   lr=0.001):
        """
        Creates the model
        :param embedding_size: Dimensionality of the Embedding
        :param lstm_units: Number of units
        :param lstm_layers: Number of stacked layers
        :param lr: Optimizer Learning rate
        :return: Sequential Model
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=len(self.vocab),
                                            output_dim=embedding_size,
                                            batch_input_shape=[self.bs, None]))
        for i in range(lstm_layers):
            model.add(tf.keras.layers.LSTM(units=lstm_units,
                                           return_sequences=True,
                                           stateful=True))
        model.add(tf.keras.layers.Dense(len(self.vocab)))
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        return model

    def train(self,
              model,
              epochs=100,
              callbacks=None):
        """
        Compiles and Trains the model using the .fit method on a Sequential model
        :param model: An object of tf.keras.models.Sequential
        :param epochs: Number of training loops
        :param callbacks: Specify callbacks as a list as how an argument is passed
        :return: None
        """

        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                   logits,
                                                                   from_logits=True)

        model.compile(optimizer=self.opt,
                      loss=loss)
        model.fit(self.train_data, epochs=epochs, callbacks=callbacks)

    def _read_file(self, file):
        """
        Reads the file passed
        :param file: File name as relative path to parent directory
        :return: None
        """
        self.original_text = open(file).read()
        self.vocab = sorted(set(self.original_text))
        self.char2idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

    def _tokenize(self):
        """
        Tokenizes the dataset into integers by the dictionary
        :return: The Int Tensor form of the data
        """
        tokenized = np.array([self.char2idx[c] for c in self.original_text])
        tensor_dataset = tf.data.Dataset.from_tensor_slices(tokenized)
        return tensor_dataset

    def _create_dataset(self, tensor_dataset):
        """
        Batches the data into sizes of 'self.seq_len + 1'
        :param tensor_dataset: The dataset in the form of Int Tensors
        :return: The batches
        """
        sequences = tensor_dataset.batch(self.seq_len + 1, drop_remainder=True)
        return sequences

    def _create_train_dataset(self, sequences):
        """
        Makes the train data and labels.
        Train data - all but last character.
        Train labels - all but first character
        :param sequences: The batches of data of sizes 'self.seq_len + 1'
        :return: None
        """
        dataset = sequences.map(self.__split_input_target)
        self.train_data = dataset.shuffle(10000).batch(self.bs, drop_remainder=True)

    @staticmethod
    def __split_input_target(chunk):
        """
        Performs the required computations for the _create_train_dataset function.
        :param chunk: A batched dataset of size 'self.seq_len + 1'
        :return: The train data and labels
        """
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
