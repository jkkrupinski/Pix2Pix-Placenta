# Pix2Pix Placenta
Repository for converting segmentation images of placenta to realistic looking ones using pix2pix network.
There are two main files avilable:
- pix2pix_placenta.ipynb in jupyter notebook format, for training network on the dataset.
- main.py for testing results from network including single image conversion and biger image convesion using stiching.

## pix2pix:
Model used for converting is avaliable in kaggle [here](https://www.kaggle.com/models/jakubkrupinski/pix2pix-placenta)

Original paper can be found [here](https://paperswithcode.com/method/pix2pix)

[Here pic of model]

## FetReg dataset:
Model was trained on modified dataset avaliable in kaggle [here](https://www.kaggle.com/datasets/jakubkrupinski/fetreg) both in .keras and .h5 formats.

Original data comes form [here](https://paperswithcode.com/dataset/fetreg)

## Usage:
Images from dataset are rgb images with black `(0,0,0)` background and segmented vessels `(1,1,1)` to better distinguish vessels form background image pixels are multipiled by 40. 

`generator()` model gets as input images of size 256x256 that are cast to `tf.float32` data type, than images are normalized and `None` batch dimension is added to fit to expected model input.

## ToDo
- format and refactor code + repo
- train on inproved dataset without lasers
- update kaggle
- finish README