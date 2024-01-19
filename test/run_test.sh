#!/bin/bash

echo "* Performing test on univariate systems..."
cp input_uni/input_parameters.txt test_uni/

cd test_uni/
bash RUN_TEST.sh > ../log_uni.txt
cd ../

cmp -s test_uni/final_states.txt output_uni/final_states.txt
cmp -s test_uni/final_thresholds.txt output_uni/final_thresholds.txt
cmp -s test_uni/fraction_0.txt output_uni/fraction_0.txt
cmp -s test_uni/number_of_states.txt output_uni/number_of_states.txt

echo "* Performing test on multivariate systems..."
cp input_multi/input_parameters.txt test_multi/

cd test_multi/
bash RUN_TEST.sh > ../log_multi.txt
cd ../

cmp -s test_multi/final_states.txt output_multi/final_states.txt
cmp -s test_multi/fraction_0.txt output_multi/fraction_0.txt
cmp -s test_multi/number_of_states.txt output_multi/number_of_states.txt