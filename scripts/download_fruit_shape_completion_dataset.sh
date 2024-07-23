#!/bin/bash

mkdir data -p
cd data

echo Downloading dataset ...
wget https://www.ipb.uni-bonn.de/html/projects/shape_completion/shape_completion_challenge.zip

echo Extracting dataset...
unzip shape_completion_challenge.zip
rm shape_completion_challenge.zip

cd ../..