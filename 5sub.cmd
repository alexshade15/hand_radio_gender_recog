#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# How much memory is needed (per node). Possible units: K, G, M, T
#SBATCH --mem=24G

# set a partition
#SBATCH --partition standard

# aaa SBATCH --export=none
#SBATCH --gres=gpu:1

# set max wallclock time
#SBATCH --time=48:00:00

# set name of job
#SBATCH --job-name=TrainVgg

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# set an output file
#SBATCH --output 5out_vgg.dat

# send mail to this address
#SBATCH --mail-user=vlongoba@uni-muenster.de
#SBATCH -v

# run the application
srun singularity exec --bind /data/vlongoba:/data /data/sifs/tensorflow2.sif python 5vgg.py
# srun singularity exec --bind /data/vlongoba:/data /data/sifs/tensorflow2.sif python jackzen.py
