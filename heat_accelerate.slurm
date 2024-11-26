#!/bin/bash
#SBATCH --job-name=heat_gpu
#SBATCH --partition=gpubase
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=heat_gpu.out

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Node: $(hostname)"

# Load required modules
module load cuda/11.0
module load gcc/9.3.0
module load openmpi/4.0.3

echo "Modules loaded"

# Compile the code
nvcc -Xcompiler -fopenmp -o heat_gpu heat_gpu.cu

# Check if executable exists
if [ -f ./heat_gpu ]; then
    echo "Found executable heat_gpu"
    ls -l ./heat_gpu
else
    echo "ERROR: heat_gpu not found"
    exit 1
fi

# Run the program
echo "Starting MPI run with GPU acceleration..."
mpirun -np 2 ./heat_gpu

echo "Job finished at $(date)"
