#!/bin/bash
#SBATCH --partition=t4v2
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --job-name=wkchen
#SBATCH --output=log.roberta.sick.snli
#SBATCH --ntasks=1

echo Running on $(hostname)
date

python runner.py --model roberta-large --dataset snli --template 1