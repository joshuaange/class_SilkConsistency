#!/bin/bash
#SBATCH --job-name=Planck_MCMC_ell_d_noLensing
#SBATCH --account=hill
#SBATCH --nodes=1
#SBATCH --ntasks=4                # 4 MPI processes = 4 MCMC chains
#SBATCH --cpus-per-task=8         # 8 CLASS threads per chain (4 x 8 = 32 cores)
#SBATCH --mem-per-cpu=500M
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out   # create the logs/ folder before submitting

module load openmpi/4.1.6-gcc11
source ~/envs/ange_cosmology/bin/activate
export LD_LIBRARY_PATH=/insomnia001/shared/apps/openmpi/4.1.6-gcc11/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Slurm: SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK, job $SLURM_JOB_ID"

cd $SLURM_SUBMIT_DIR

mpirun -np $SLURM_NTASKS --bind-to none -x OMP_NUM_THREADS cobaya-run Planck_MCMC_ell_d_noLensing.yaml -p ../packages --resume

# Run with 
#   - source ~/envs/ange_cosmology/bin/activate
#   - sbatch run_Planck_MCMC_ell_d_noLensing.sh