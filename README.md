# Parallel Heat Equation Solver with GPU Acceleration

## Overview

This project implements a parallel solution to the 2D heat equation using a hybrid approach combining MPI, OpenMP, and CUDA. The solver includes both CPU-only and GPU-accelerated versions, demonstrating high-performance computing techniques for numerical simulations.

## Requirements

- GCC 9.3.0 or later
- OpenMPI 4.0.3
- CUDA 11.0 (for GPU version)
- SLURM workload manager

## Building the Project

### CPU Version

```bash
module load gcc/9.3.0
module load openmpi/4.0.3
mpicc -fopenmp -o heat_parallel heat_parallel.c
```

### GPU Version

```bash
module load cuda/11.0
module load gcc/9.3.0
module load openmpi/4.0.3
nvcc -Xcompiler -fopenmp -o heat_accelerate heat_accelerate.cu
```

## Running the Simulation

### CPU Version

```bash
sbatch heat_job.slurm
```

### GPU Version

```bash
sbatch heat_accelerate.slurm
```

## Problem Configuration

- Grid size: 500 × 500
- Maximum iterations: 1000
- Convergence tolerance: 1e-6
- Boundary conditions: Fixed at 100.0
