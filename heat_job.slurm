#!/bin/bash
#SBATCH --job-name=heat_parallel
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:30:00
#SBATCH --output=heat_parallel.out
#SBATCH --nodelist=node1    

echo "Job started at $(date)"
echo "Working directory: $(pwd)"
echo "Node: $(hostname)"

# Change to the directory containing your executable
cd /home/user065/hw
echo "Changed to directory: $(pwd)"

# Load required modules
echo "Loading modules..."
module load gcc/9.3.0
module load openmpi/4.0.3
echo "Modules loaded"

# Check if executable exists
if [ -f ./heat_parallel ]; then
    echo "Found executable heat_parallel"
    ls -l ./heat_parallel
else
    echo "ERROR: heat_parallel not found"
    exit 1
fi

# Run the program
echo "Starting MPI run..."
mpirun -np 2 ./heat_parallel
echo "Job finished at $(date)"
