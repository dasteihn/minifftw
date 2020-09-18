#!/bin/bash

#SBATCH -J one-node-one-task-test
#SBATCH -D ./
#SBATCH -o ./logfiles/%x.%j.%N.out
#SBATCH --clusters=cm2_tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --get-user-env
#SBATCH --mail-type=end
#SBATCH --mem=5000mb
#SBATCH --mail-user=stanner@posteo.de
#SBATCH --export=NONE
#SBATCH --time=8:15:00

module load slurm_setup
module load fftw/3.3.8-intel19-impi
module load python/3.6_intel
source activate py38
time mpiexec -n $SLURM_NTASKS python3 tests/perf_test.py
