#!/bin/bash
#SBATCH --job-name=Mock_LCDM_MCMC_eps_d
#SBATCH --account=hill
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=500M
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
module load openmpi/4.1.6-gcc11
source ~/envs/ange_cosmology/bin/activate
export LD_LIBRARY_PATH=/insomnia001/shared/apps/openmpi/4.1.6-gcc11/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $SLURM_SUBMIT_DIR
mpirun -np $SLURM_NTASKS --bind-to none --mca btl ^openib -x OMP_NUM_THREADS cobaya-run Mock_LCDM_MCMC_eps_d.yaml -p ../../packages --resume
