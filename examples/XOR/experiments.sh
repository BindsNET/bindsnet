#!/usr/bin/env bash

for seed in {0..999}
do
    sbatch get_learning_curve.sh $seed
done