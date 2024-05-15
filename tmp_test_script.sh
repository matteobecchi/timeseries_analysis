#!/bin/bash

cd test

pytest test_multi.py

rm -rf all_labels.npy data_directory.txt input_parameters.txt
rm -rf output_figures/*_Fig1_*.png
rm -rf output_figures/*_Fig0.png

cd ../
