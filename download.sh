#!/bin/bash

# This script will download shrec13 and shrec14 to dataset folder
# This script will also download the retrieval list to returnList folder


if [ -d "dataset" ]; then
    mkdir dataset
fi

echo "Download SHREC13 feature"
wget -l 100 --user=guest --password=guest ftp://10.224.82.6/ftp/shrec13feature.tar.gz -P dataset
echo "Download SHREC13 list file"
wget -l 100 --user=guest --password=guest ftp://10.224.82.6/ftp/shrec13_shapeList.txt -P dataset
wget -l 100 --user=guest --password=guest ftp://10.224.82.6/ftp/shrec13_trainList.txt -P dataset
wget -l 100 --user=guest --password=guest ftp://10.224.82.6/ftp/shrec13_testList.txt -P dataset

echo "Download SHREC14 feature"
wget -l 100 --user=guest --password=guest ftp://10.224.82.6/ftp/shrec14feature.tar.gz -P dataset

echo "Download SHREC14 list file"
wget -l 100 --user=guest --password=guest ftp://10.224.82.6/ftp/shrec14_shapeList.txt -P dataset
wget -l 100 --user=guest --password=guest ftp://10.224.82.6/ftp/shrec14_trainList.txt -P dataset
wget -l 100 --user=guest --password=guest ftp://10.224.82.6/ftp/shrec14_testList.txt -P dataset

echo "extracting file"

tar -xvf ./dataset/shrec13feature.tar.gz  -C dataset
tar -xvf ./dataset/shrec14feature.tar.gz  -C dataset

