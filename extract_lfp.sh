#!/bin/bash


#PBS -N Extracting 

#PBS -l nodes=2:ppn=4:gpus=1,walltime=03:00:00

#module load cuda-9.0p1
#module load cuDNN-7.1.1

. /scratch/khengen_lab/anaconda3/etc/profile.d/conda.sh

python /home/ltilden/extract_lfp.py &> /home/ltilden/extractlfp.log
