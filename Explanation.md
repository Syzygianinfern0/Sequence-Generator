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
The `TOKENIZED`<sup id="a1">[1](#f1)</sup> variable is created by converting the whole 
`TEXT`<sup id="a2">[2](#f2)</sup> into integers by 
using the dictionary for conversion.

<b id="f1">1</b> Shape: {None} Type: str [↩](#a1)

<b id="f2">2</b> Shape: {None} Type: ndarray [↩](#a2)

## Datasets
### Basic
The entire 



