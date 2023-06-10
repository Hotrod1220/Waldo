## Where's Waldo?

A project for CPSC 3750: Artificial Intelligence using PyTorch. We created a dataset from six different images of Waldo where he was augmented and placed onto a random color background or historical art background (which consisted of 54,020 images). We aimed to create a dataset of at least 60,000+ images that were 224x224 in size.

Afterwards, we created a smaller testing dataset from a "Where's Waldo?" influenced wallpaper where Waldo was resized and placed in a random location for each image.


## Source

### Historical Artwork
* https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time

* https://www.kaggle.com/datasets/ansonnnnn/historic-art


## Prerequisites

* [pyenv](https://github.com/pyenv/pyenv) or [Python 3.11.2](https://www.python.org/downloads/)


## Setup

### pyenv

```
pyenv install 3.11.2
```

```
pyenv local 3.11.2
```

### Virtual Environment

```
python -m venv venv
```

#### Windows

```
"venv/Scripts/activate"
```

#### Unix

```
source venv/bin/activate
```

### Packages

```
pip install -U -r requirements.txt
```
