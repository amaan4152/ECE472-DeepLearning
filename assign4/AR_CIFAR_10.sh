#!/bin/bash
#
#SBATCH --job-name=AR_CIFAR_10_EXP
#SBATCH --output=/zooper2/amaan.rahman/ECE472-DeepLearning/assign4/output.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=1gb

export HOME=/zooper2/amaan.rahman
python3 /zooper2/amaan.rahman/ECE472-DeepLearning/assign4/cifar_class.py
