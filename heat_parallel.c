#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NX 500
#define NY 500
#define MAX_ITER 1000
#define TOLERANCE 1e-6

double max_double(double a, double b) { return (a > b) ? a : b; }

void initialize_grid(double *u, int start_row, int num_rows, int ny) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < ny; j++) {
      u[i * ny + j] = 0.0;
      if (start_row + i == 0 || start_row + i == NX - 1 || j == 0 ||
          j == NY - 1) {
        u[i * ny + j] = 100.0; // Boundary conditions
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int rank, size, provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    printf("Starting simulation with %d MPI processes\n", size);
    printf("Grid size: %d x %d\n", NX, NY);
  }

  // Calculate local grid dimensions
  int rows_per_proc = NX / size;
  int start_row = rank * rows_per_proc;
  int num_rows = (rank == size - 1) ? (NX - start_row) : rows_per_proc;
  int local_size = (num_rows + 2) * NY; // +2 for ghost rows

  if (rank == 0) {
    printf("Rows per process: %d\n", rows_per_proc);
  }
  printf("Process %d handling rows %d to %d\n", rank, start_row,
         start_row + num_rows - 1);

  // Allocate local arrays
  double *u = (double *)malloc(local_size * sizeof(double));
  double *u_new = (double *)malloc(local_size * sizeof(double));

  if (u == NULL || u_new == NULL) {
    printf("Error: Memory allocation failed on process %d\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  initialize_grid(u, start_row, num_rows, NY);
  printf("Process %d initialized grid\n", rank);

  // Create MPI datatypes for ghost exchange
  MPI_Datatype row_type;
  MPI_Type_contiguous(NY, MPI_DOUBLE, &row_type);
  MPI_Type_commit(&row_type);

  double max_diff, global_max_diff;
  int iter;

  // Main iteration loop
  for (iter = 0; iter < MAX_ITER; iter++) {
    // Exchange ghost rows
    if (rank > 0) {
      MPI_Sendrecv(&u[NY], 1, row_type, rank - 1, 0, &u[0], 1, row_type,
                   rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
      MPI_Sendrecv(&u[(num_rows - 1) * NY], 1, row_type, rank + 1, 0,
                   &u[(num_rows)*NY], 1, row_type, rank + 1, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    }

    max_diff = 0.0;

// Compute new values
#pragma omp parallel
    {
      double local_max_diff = 0.0;

#pragma omp for collapse(2)
      for (int i = 1; i < num_rows + 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
          u_new[i * NY + j] =
              0.25 * (u[(i + 1) * NY + j] + u[(i - 1) * NY + j] +
                      u[i * NY + j + 1] + u[i * NY + j - 1]);
          double diff = fabs(u_new[i * NY + j] - u[i * NY + j]);
          local_max_diff = max_double(local_max_diff, diff);
        }
      }

#pragma omp critical
      {
        max_diff = max_double(max_diff, local_max_diff);
      }
    }

// Update u
#pragma omp parallel for collapse(2)
    for (int i = 1; i < num_rows + 1; i++) {
      for (int j = 1; j < NY - 1; j++) {
        u[i * NY + j] = u_new[i * NY + j];
      }
    }

    // Check for convergence across all processes
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

  // Output final state information
  if (rank == 0) {
    if (iter == MAX_ITER) {
      printf("Did not converge after %d iterations. Final max difference: %e\n",
             MAX_ITER, global_max_diff);
    }
    printf("Simulation completed.\n");
  }

  // Clean up
  MPI_Type_free(&row_type);
  free(u);
  free(u_new);
  MPI_Finalize();
  return 0;
}
