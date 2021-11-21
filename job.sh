#!/bin/bash
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=96GB
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -l wd
#PBS -l jobfs=100GB
#PBS -l storage=scratch/sz65
#PBS -P jk87

source ~/pytorch-semseg/bin/activate
cd /scratch/jk87/rh5755/trajectory_prediction/sgan_raw
module load python3/3.9.2
module load pytorch/1.9.0
module load cudnn/8.1.1-cuda11

sh train_univ_12.sh ccattention/seperate_hw_activation
