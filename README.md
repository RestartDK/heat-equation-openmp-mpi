# Parallel Implementation of Heat Equation Solver

## Analysis Report

### Introduction

This report discusses the implementation of a parallel heat equation solver using a hybrid MPI/OpenMP approach. The project aimed to optimize a sequential heat equation solver by leveraging both distributed and shared memory parallelism, utilizing the available cluster resources effectively.

### Parallelization Strategy

The parallelization strategy employed a two-level approach combining MPI for distributed memory parallelism and OpenMP for shared memory parallelism. For the MPI implementation, the 2D grid (500×500) was decomposed by rows among multiple processes. Each MPI process was assigned a contiguous block of rows to handle, with ghost rows implemented for boundary exchanges between processes. This domain decomposition strategy was chosen to minimize communication overhead while maintaining load balance.

Within each MPI process, OpenMP was utilized to further parallelize the computational loops. The implementation leveraged OpenMP's collapse directive for nested loops, ensuring efficient utilization of shared memory resources. Special attention was paid to thread safety, particularly in the reduction operations for calculating maximum differences during the convergence check.

### Implementation Challenges and Solutions

The most significant challenges encountered during this project centered around cluster resource management and job submission. One particularly noteworthy challenge emerged during the initial attempts to run the parallel implementation. Upon submitting the job, there was considerable confusion when the job wouldn't execute immediately. After investigation and learning to use various SLURM commands (sinfo, squeue), it became apparent that all compute nodes were allocated to other users' jobs, showing "alloc" status.

Another significant challenge arose from node assignment issues. Initial job submissions failed with a "ReqNodeNotAvail" error because the script was attempting to use unavailable nodes, specifically attempting to access gpu-node1 which was in a down state. This was resolved by carefully modifying the SLURM script to correctly specify available nodes and appropriate resource requests. The solution involved using the correct partition (cpubase_bycore_b1) and requesting a reasonable number of nodes and tasks based on actual availability.

### Resource Management Strategy

Through these challenges, a systematic approach to resource management evolved. Job submissions were modified to start with conservative resource requests, improving the likelihood of job allocation. The SLURM script was refined to include specific partition requests and appropriate task counts. This approach proved more successful than initial attempts at maximizing resource usage.

The learning process established several best practices for cluster usage. Regular monitoring of resource availability became standard practice before job submission. Job tracking was implemented using specific job names and regular status checks. Resource requests were carefully matched to actual computational needs while considering queue times and system availability.

### Performance Analysis

The performance analysis of our parallel heat equation solver revealed several significant insights into both the numerical behavior and computational efficiency of the implementation. Running on a single node with 2 MPI processes, the simulation completed in approximately 4 seconds for a 500×500 grid problem size. While this represents reasonable performance for the given configuration, the results highlighted several areas for optimization and improvement.

Our execution data showed interesting convergence behavior across 1000 iterations. The solver exhibited an initial rapid change in the maximum difference, starting at 5.0e+01 and stabilizing around 2.41e+01. However, the simulation did not achieve the desired convergence tolerance of 1e-6 within the maximum iteration limit. This behavior suggests that while our parallel implementation successfully distributed the computational load, the numerical properties of our solver require further refinement.

### Conclusion

The development of this parallel heat equation solver provided valuable insights into both technical implementation challenges and practical aspects of high-performance computing environments. While the technical aspects of parallel programming presented their own challenges, the experience of working with a shared cluster environment proved equally educational. The project highlighted the importance of understanding not just the computational aspects but also the operational environment in which parallel programs execute.

The lessons learned about resource management and job scheduling will prove valuable for future high-performance computing projects. Understanding the relationship between resource requests, queue times, and system availability has become as crucial as the parallel programming techniques themselves.
