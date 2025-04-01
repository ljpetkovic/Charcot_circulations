#!/bin/bash

#SBATCH --job-name=metadata
#SBATCH --output=metadata.out
#SBATCH --mail-user=ljudmila.petkovic@sorbonne-universite.fr
#SBATCH --mail-type=ALL
#SBATCH --partition=std
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## Load software modules
module load python/3.10.14

python3 metadata.py



## Move data (if necessary) and launch application
## (use 'srun' for launching mpi applications)


## Finish gracefully
exit 0
