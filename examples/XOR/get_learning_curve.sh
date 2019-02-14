#!/usr/bin/env bash
#
#SBATCH --job-name=mstdp
#SBATCH --partition=defq
#SBATCH --time=00-00:10:00
#SBATCH --mem=1000
#SBATCH --account=rkozma
#SBATCH --output=../../../output/mstdp_%j.out
#SBATCH -e ../../../output/error_%j.err
#SBATCH --cpus-per-task=8

seed=${1:-0}

echo $seed $n_train $n_test $time $lr $lr_decay $update_interval $max_prob

python mstdp.py --seed $seed

exit