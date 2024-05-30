#!/bin/bash
# same as https://github.com/PRBonn/TCoRe

mkdir data -p
cd data

echo Downloading dataset ...
wget -O data.zip -c https://www.ipb.uni-bonn.de/html/projects/shape_completion/igg_fruit.zip

echo Extracting dataset...
unzip data.zip
rm data.zip

cd ../..