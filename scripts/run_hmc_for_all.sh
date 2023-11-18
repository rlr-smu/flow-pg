#!/bin/bash
for count in 10000 1000000; do
    echo "Running for ${count}"
    python -m experiments.generate_samples_with_hmc --problems All --count $count --log_dir logs/sample_generation/$count
done