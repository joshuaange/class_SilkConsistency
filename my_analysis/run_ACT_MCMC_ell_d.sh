#!/bin/bash
#SBATCH --job-name=ACT_MCMC_ell_d
#SBATCH --account=hill
#SBATCH --nodes=1
#SBATCH --ntasks=4                # 4 MPI processes = 4 MCMC chains
#SBATCH --cpus-per-task=8         # 8 CLASS threads per chain
#SBATCH --mem-per-cpu=500M        # ~16 GB total; the chains need only a few GB
#SBATCH --time=4-00:00:00         # ACT needs CLASS up to ell = 9000, so this run is slower
#SBATCH --output=logs/%x_%j.out

module load openmpi/4.1.6-gcc11
source ~/envs/ange_cosmology/bin/activate
export LD_LIBRARY_PATH=/insomnia001/shared/apps/openmpi/4.1.6-gcc11/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Slurm: SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK, job $SLURM_JOB_ID"

cd $SLURM_SUBMIT_DIR

mpirun -np $SLURM_NTASKS --bind-to none --mca btl ^openib -x OMP_NUM_THREADS \
    cobaya-run ACT_MCMC_ell_d.yaml -p ../packages --resume

# Run with 
#   - source ~/envs/ange_cosmology/bin/activate
#   - sbatch run_ACT_MCMC_ell_d.sh