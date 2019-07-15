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
The `TOKENIZED`[^1] variable is created by converting the whole `TEXT`[^2] into integers by 
using the dictionary for conversion.

[^1]: Shape: {None} Type: str
[^2]: Shape: {None} Type: ndarray

## Datasets
### Basic
The entire 


