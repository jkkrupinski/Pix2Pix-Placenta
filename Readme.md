# Repository for converting segmentation images of placenta to realistic looking ones using pix2pix

# pix2pix:
[here](https://paperswithcode.com/method/pix2pix)

# FetReg dataset:
[here](https://paperswithcode.com/dataset/fetreg)

# Usage:
Model used for converting is avaliable in kaggle [here](https://www.kaggle.com/models/jakubkrupinski/pix2pix-placenta)

Model was trained on modified dataset avaliable in kaggle [here](https://www.kaggle.com/datasets/jakubkrupinski/fetreg)

Images from dataset are rgb images with black `(0,0,0)` background and segmented vessels `(1,1,1)` to better distinguish vessels form background image pixels are multipiled by 40. `generator()` model gets as input images of size 256x256 that are cast to `tf.float32` data type, than images are normalized and `None` batch dimension is added to fit to expected model input.