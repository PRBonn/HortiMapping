#!/bin/bash

mkdir data -p
cd data

echo Downloading dataset ...
wget -O BUP20_example_data.zip -c https://uni-bonn.sciebo.de/s/ovg3hIXHOeHdht6/download

echo Extracting dataset...
unzip BUP20_example_data.zip
rm BUP20_example_data.zip

cd ../..