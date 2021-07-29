# An Ensemble Learning-Based No Reference Qoe Model For User Generated Contents

## Overview

This repository contains the source codes of our proposed QoE model for User-generated Content Videos described in the following publication:

``
Duc Nguyen, H. Tran and T. Thang,  "An Ensemble Learning-Based No Reference Qoe Model For User Generated Contents," in 2021 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), Shenzhen, China, 2021 pp. 1-6. doi: 10.1109/ICMEW53276.2021.9455959
``

To use our code, first following requirements.txt file to setup the enviroment.

## Download dataset

Execute file ``dataset/download_video.sh`` to download the dataset that contains 7200 UGC videos. The total data size is ~35GB, thus it might take up to a day to download all the videos.
```
$ cd dataset
$ bash download_video.sh
```

Divide the download videos into ``dataset/train`` folder and ``dataset/test`` folder using information in ``train_label.csv`` and ``test_label.csv`` 

## Feature extraction
To extract features from the videos, run the following command from the terminal.
```
$ python3 feature_extractor.py
```
Extracted video features will be saved to ``dataset/train_feature.csv`` and ``dataset/test_feature.csv``

## Training
To train our model, run the following command from the terminal.

```
$ python3 train.py
```

The trained model will be saved to ``pretrained/model.pkl``

## Testing

Prepare a folder containing test videos, then run the following command.

```
$ python3 prediction.py /path/to/test/folder
```

Prediction results will be saved to a file named 'test_result.csv' in the current folder.
