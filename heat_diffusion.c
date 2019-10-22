#include "helper.h"
#include "mpi.h"

int
main(int argc, char* argv[])
{
  MPI_Init(NULL, NULL);
  int nmb_mpi_proc, mpi_rank;
  MPI_Comm mpiworld = MPI_COMM_WORLD;

  MPI_Comm_size(mpiworld, &nmb_mpi_proc);
  MPI_Comm_rank(mpiworld, &mpi_rank);

  int niter = 0;
  double c = 0.0;
  char* ptr1 = NULL;
  char* ptr2 = NULL;
  if (argc == 3) {
    for (size_t ix = 1; ix < argc; ++ix) {
      ptr1 = strchr(argv[ix], 'n');
      ptr2 = strchr(argv[ix], 'd');
      if (ptr1) {
        niter = strtol(++ptr1, NULL, 10);
      } else if (ptr2) {
        c = atof(++ptr2);
      }
    }
  } else {
    printf("Invalid number of arguments. Correct syntax is: heat_diffusion "
           "-n#numberOfTimeSteps4 -d#diffusionConstant\n");
    return 1;
  }
  const double c1 = 1 - c;
  const double c0 = c;

  int nx, ny;
  char line[80];
  FILE* input;
  if (mpi_rank == 0) {
    input = fopen("diffusion", "r");
    if (input == NULL) {
      perror("Error opening file");
      return 2;
    }
    fgets(line, sizeof(line), input);
    sscanf(line, "%d %d", &ny, &nx);
  }
  MPI_Bcast(&nx, 1, MPI_INT, 0, mpiworld);
  MPI_Bcast(&ny, 1, MPI_INT, 0, mpiworld);
  int nx2 = nx + 2;
  int ny2 = ny + 2;
  int nx2ny2 = (nx2) * (ny2);

  size_t const ix_sub_m = (nx - 1) / nmb_mpi_proc + 1;
  int lens[nmb_mpi_proc];
  int poss[nmb_mpi_proc];
  if (mpi_rank == 0) {
    for (size_t jx = 0, pos = 0; jx < nmb_mpi_proc; ++jx, pos += ix_sub_m) {
      lens[jx] = ix_sub_m < nx - pos ? ix_sub_m : nx - pos;
      poss[jx] = pos + 1;
    }
  }
  MPI_Bcast(lens, nmb_mpi_proc, MPI_INT, 0, mpiworld);
  MPI_Bcast(poss, nmb_mpi_proc, MPI_INT, 0, mpiworld);
  int loclen = lens[mpi_rank];
  int lastNodeRnk = nmb_mpi_proc - 1;

  double* ivEntriesFull;
  double* ivEntries;
  double* nextTimeEntries;
  if (mpi_rank == 0) {
    ivEntriesFull = (double*)calloc(sizeof(double), nx2ny2);
  }
  ivEntries = (double*)calloc(sizeof(double), (loclen + 2) * ny2);
  nextTimeEntries = (double*)calloc(sizeof(double), (loclen + 2) * ny2);

  if (mpi_rank == 0) {
    int i, j;
    double t;
    while (fgets(line, sizeof(line), input) != NULL) {
      sscanf(line, "%d %d %lf", &j, &i, &t);
      ivEntriesFull[(i + 1) * (ny + 2) + j + 1] = t;
    }
    fclose(input);

    memcpy(ivEntries, ivEntriesFull, sizeof(double) * (lens[0] + 2) * ny2);

    if (nmb_mpi_proc > 1) {
      for (size_t ix = 1; ix < nmb_mpi_proc; ix++) {
        MPI_Send(ivEntriesFull + (poss[ix] - 1) * ny2,
                 (lens[ix] + 2) * ny2,
                 MPI_DOUBLE,
                 ix,
                 0,
                 mpiworld);
      }
    }
  }
  if (mpi_rank > 0) {
    MPI_Recv(ivEntries, (loclen + 2) * ny2, MPI_DOUBLE, 0, 0, mpiworld, NULL);
  }

  double sum = 0.0;
  for (int n = 0; n < niter; ++n) {
    // strtim = MPI_Wtime();
    for (int ix = 1; ix < loclen + 1; ++ix) {
      for (int jx = 1; jx < ny + 1; ++jx) {

        double hij = ivEntries[ix * (ny2) + jx];
        double hijW = ivEntries[ix * (ny2) + jx - 1];
        double hijE = ivEntries[ix * (ny2) + jx + 1];
        double hijS = ivEntries[(ix + 1) * (ny2) + jx];
        double hijN = ivEntries[(ix - 1) * (ny2) + jx];

        nextTimeEntries[ix * (ny2) + jx] =
          hij * c1 + 0.25 * c0 * (hijW + hijE + hijS + hijN);
        if (n + 1 == niter) {
          sum += nextTimeEntries[ix * (ny2) + jx];
        }
      }
    }

    double* swp = nextTimeEntries;
    nextTimeEntries = ivEntries;
    ivEntries = swp;

    if (nmb_mpi_proc > 1) {
      // messaging from rank 0 to rank n
      if (mpi_rank == 0) {
        // send the curren rank last line to the right rank
        double* sendind = ivEntries + (loclen) * (ny2);
        MPI_Send(sendind, ny2, MPI_DOUBLE, 1, 0, mpiworld);
        // receive right rank top line to the curren rank btm - 1
        double* recvind = ivEntries + (loclen + 1) * (ny2);
        MPI_Recv(recvind, ny2, MPI_DOUBLE, 1, 0, mpiworld, NULL);
      } else if (mpi_rank == lastNodeRnk) {

        // receive left rank last line to the curren rank top - 1
        double* recvind = ivEntries;
        MPI_Recv(recvind, ny2, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld, NULL);
        // send the curren rank top line to the left rank
        double* sendind = ivEntries + (ny2);
        MPI_Send(sendind, ny2, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld);
      } else {

        // receive left rank last line to the curren rank top - 1
        double* recvindl2r = ivEntries;
        MPI_Recv(recvindl2r, ny2, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld, NULL);
        // send the curren rank last line to the right rank
        double* sendindl2r = ivEntries + loclen * (ny2);
        MPI_Send(sendindl2r, ny2, MPI_DOUBLE, mpi_rank + 1, 0, mpiworld);

        // send the curren rank top line to the right rank
        double* sendindr2l = ivEntries + (ny2);
        MPI_Send(sendindr2l, ny2, MPI_DOUBLE, mpi_rank - 1, 0, mpiworld);
        // receive right rank top line to the curren rank btm - 1
        double* recvindr2l = ivEntries + (loclen + 1) * (ny2);
        MPI_Recv(recvindr2l, ny2, MPI_DOUBLE, mpi_rank + 1, 0, mpiworld, NULL);
      }
    }
  }

  double redSum = 0.0;
  MPI_Reduce(&sum, &redSum, 1, MPI_DOUBLE, MPI_SUM, 0, mpiworld);
  double avg = redSum / (double)(nx * ny);
  MPI_Bcast(&avg, 1, MPI_DOUBLE, 0, mpiworld);
  double abssum = 0.0;
  for (int ix = 1; ix < loclen + 1; ++ix) {
    for (int jx = 1; jx < ny + 1; ++jx) {
      double absdelta = ivEntries[ix * (ny2) + jx] - avg;
      absdelta = (absdelta < 0.0 ? -absdelta : absdelta);
      abssum += absdelta;
    }
  }

  double redAbsSum = 0.0;
  MPI_Reduce(&abssum, &redAbsSum, 1, MPI_DOUBLE, MPI_SUM, 0, mpiworld);
  double absavg = redAbsSum / (double)(nx * ny);

  if (mpi_rank == 0) {
    free(ivEntriesFull);
  }
  free(nextTimeEntries);
  free(ivEntries);

  if (mpi_rank == 0) {
    printf("average: %e\n", avg);
    printf("average absolute difference: %e\n", absavg);
  }

  MPI_Finalize();
  return 0;
}
