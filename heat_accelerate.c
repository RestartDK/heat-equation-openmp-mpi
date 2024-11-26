#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NX 500
#define NY 500
#define MAX_ITER 1000
#define TOLERANCE 1e-6
#define BLOCK_SIZE 16

// CUDA kernel for heat equation update
__global__ void heat_kernel(double *u, double *u_new, int num_rows, int ny,
                            double *max_diff) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

  extern __shared__ double shared_max[];
  double local_max = 0.0;

  if (i < num_rows + 1 && j < ny - 1) {
    u_new[i * ny + j] = 0.25 * (u[(i + 1) * ny + j] + u[(i - 1) * ny + j] +
                                u[i * ny + j + 1] + u[i * ny + j - 1]);
    double diff = fabs(u_new[i * ny + j] - u[i * ny + j]);
    local_max = diff;
  }

  // Reduction in shared memory
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  shared_max[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_max[tid] = fmax(shared_max[tid], shared_max[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicMax((unsigned long long int *)max_diff,
              __double_as_longlong(shared_max[0]));
  }
}

void initialize_grid(double *u, int start_row, int num_rows, int ny) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < num_rows + 2; i++) {
    for (int j = 0; j < ny; j++) {
      u[i * ny + j] = 0.0;
      if (start_row + i - 1 == 0 || start_row + i - 1 == NX - 1 || j == 0 ||
          j == NY - 1) {
        u[i * ny + j] = 100.0;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int rank, size, provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Calculate local grid dimensions
  int rows_per_proc = NX / size;
  int start_row = rank * rows_per_proc;
  int num_rows = (rank == size - 1) ? (NX - start_row) : rows_per_proc;
  int local_size = (num_rows + 2) * NY;

  // Allocate host memory
  double *u_host = (double *)malloc(local_size * sizeof(double));
  double *u_new_host = (double *)malloc(local_size * sizeof(double));
  initialize_grid(u_host, start_row, num_rows, NY);

  // Allocate device memory
  double *u_dev, *u_new_dev;
  double *max_diff_dev;
  cudaMalloc(&u_dev, local_size * sizeof(double));
  cudaMalloc(&u_new_dev, local_size * sizeof(double));
  cudaMalloc(&max_diff_dev, sizeof(double));

  // Set up CUDA grid and blocks
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((NY + block.x - 1) / block.x, (num_rows + block.y - 1) / block.y);

  double max_diff, global_max_diff;
  int iter;

  MPI_Datatype row_type;
  MPI_Type_contiguous(NY, MPI_DOUBLE, &row_type);
  MPI_Type_commit(&row_type);

  for (iter = 0; iter < MAX_ITER; iter++) {
    // Copy data to device
    cudaMemcpy(u_dev, u_host, local_size * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemset(max_diff_dev, 0, sizeof(double));

    // Exchange ghost rows using MPI
    if (rank > 0) {
      MPI_Sendrecv(&u_host[NY], 1, row_type, rank - 1, 0, &u_host[0], 1,
                   row_type, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
      MPI_Sendrecv(&u_host[(num_rows)*NY], 1, row_type, rank + 1, 0,
                   &u_host[(num_rows + 1) * NY], 1, row_type, rank + 1, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Launch kernel
    heat_kernel<<<grid, block, block.x * block.y * sizeof(double)>>>(
        u_dev, u_new_dev, num_rows, NY, max_diff_dev);

    // Copy results back
    cudaMemcpy(u_new_host, u_new_dev, local_size * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_diff, max_diff_dev, sizeof(double), cudaMemcpyDeviceToHost);

// Update solution
#pragma omp parallel for collapse(2)
    for (int i = 1; i < num_rows + 1; i++) {
      for (int j = 1; j < NY - 1; j++) {
        u_host[i * NY + j] = u_new_host[i * NY + j];
      }
    }

    // Check convergence
    MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (rank == 0 && iter % 100 == 0) {
      printf("Iteration %d, max difference: %e\n", iter, global_max_diff);
    }

    if (global_max_diff < TOLERANCE) {
      if (rank == 0) {
        printf("Converged after %d iterations.\n", iter + 1);
      }
      break;
    }
  }

  // Cleanup
  cudaFree(u_dev);
  cudaFree(u_new_dev);
  cudaFree(max_diff_dev);
  free(u_host);
  free(u_new_host);
  MPI_Type_free(&row_type);
  MPI_Finalize();
  return 0;
}
