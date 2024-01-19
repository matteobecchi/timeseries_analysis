#!/bin/bash

python3 ../../src/onion_clustering/main.py 

rm -f output_figures/*_Fig*.png
rm -f colored_trj.xyz
rm -f signal_with_labels.dat
