#!/bin/bash

wget -P . http://images.cocodataset.org/zips/train2017.zip
wget -P . http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip ./train2017.zip
unzip ./annotations_trainval2017.zip
