# Introduction
This notebook is a walkthrough for the code in the project. 
This project aims to generate text in the form as authors write them.
## Libraries Required
* `Tensorflow==2.0.0b1`
* `Numpy==1.16.4`
## CUDA Errors workaround
```python
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
```
This makes sure that the code does not hog up the GPU. It makes the code stabler and
avoid random CUDA related errors
## Vocabulary
To form the vocab to train on, we use the ```sorted(set(TEXT))```. This allows each
char to appear only once and then we sort it to maintain uniformity. A *to* and *from*
dictionary is made to allow easy conversion.
<br>
The `TOKENIZED` variable is created by converting the whole `TEXT` into integers by 
using the dictionary for conversion.

> TOKENIZED - Shape: {None} Type: ndarray <br>
  TEXT - Shape: {None} Type: str
## Datasets
### Conversion
The entire dataset is converted into a `tf.data.Dataset` as the methods associated are
very helpful
```python 
TENSOR_DATASET = tf.data.Dataset.from_tensor_slices(TOKENIZED)
```
> TENSOR_DATASET - Shape: {None,} Type: TensorSliceDataset
### Sequences
We will pack the array into sub-arrays of 101 elements each. We inout the first 100 and
have the last 100 as the labels
```python
SEQUENCES = TENSOR_DATASET.batch(seq_length + 1, drop_remainder=True)
```
> SEQUENCES - Shape: {None, 100} Type: BatchDataset
### Splitting
We will create the labels and input features here.
```python
def split_input_target(chunk):
    input_text = chunk[:-1]     # All but last
    target_text = chunk[1:]     # All but first
    return input_text, target_text
DATASET = SEQUENCES.map(split_input_target)
```
Our `DATASET` has data in the form of (input_text, target_text) if called as a `take` 
method
### Shuffling and Batching
```python
DATASET = DATASET.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
```
Our data is now packed in the form of {None, ((64,100), (64,100))
## Model
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab),             # Different ints that can be passed
                              output_dim=256,                   # Dimensionality of the Embedding
                              batch_input_shape=[64, None]),    # Batch shape for the stateful to work on
    tf.keras.layers.LSTM(units=1024,                            
                         return_sequences=True,                 # As we will be predicting from them
                         stateful=True),                        # Makes the output of y<t-1> to be the input of x<t>
    tf.keras.layers.Dense(len(vocab))                           # Returns an array of logits in the same format as input array
])
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (64, None, 256)           17408     
_________________________________________________________________
lstm (LSTM)                  (64, None, 1024)          5246976   
_________________________________________________________________
dense (Dense)                (64, None, 68)            69700     
=================================================================
Total params: 5,334,084
Trainable params: 5,334,084
Non-trainable params: 0
```

### Loss function
```python
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           logits,
                                                           from_logits=True)    # As the returned logits 
                                                                                # aren't normalised
```

## Sampling
```python
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
```
 



