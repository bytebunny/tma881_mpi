#include "helper.h"
#include "mpi.h"

int
main(int argc, char* argv[])
{
  MPI_Init(NULL, NULL);
  // double strtim, endtim, tiktork;
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
  // strtim = MPI_Wtime();

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
  const int nx2 = nx + 2;
  const int ny2 = ny + 2;
  const int nx2ny2 = (nx2) * (ny2);

  const int ix_sub_m = (nx - 1) / nmb_mpi_proc + 1;
  int lens[nmb_mpi_proc];
  int poss[nmb_mpi_proc];
  if (mpi_rank == 0) {
    for (int jx = 0, pos = 0; jx < nmb_mpi_proc; ++jx, pos += ix_sub_m) {
      lens[jx] = ix_sub_m < nx - pos ? ix_sub_m : nx - pos;
      poss[jx] = pos + 1;
    }
  }
  MPI_Bcast(lens, nmb_mpi_proc, MPI_INT, 0, mpiworld);
  MPI_Bcast(poss, nmb_mpi_proc, MPI_INT, 0, mpiworld);
  const int loclen = lens[mpi_rank];
  const int lastNodeRnk = nmb_mpi_proc - 1;

  double* ivEntriesFull;
  double* ivEntries;
  double* nextTimeEntries;
  if (mpi_rank == 0) {
    ivEntriesFull = (double*)calloc(sizeof(double), nx2ny2);
  }
  ivEntries = (double*)aligned_alloc(64, sizeof(double) * (loclen + 2) * ny2);
  nextTimeEntries =
    (double*)aligned_alloc(64, sizeof(double) * (loclen + 2) * ny2);

  // read the first line
  if (mpi_rank == 0) {
    // store initial values (note: row major order)
    int i, j;
    double t;
    while (fgets(line, sizeof(line), input) != NULL) {
      sscanf(line, "%d %d %lf", &j, &i, &t);
      ivEntriesFull[(i + 1) * (ny + 2) + j + 1] = t;
    }
    fclose(input);

    memcpy(ivEntries, ivEntriesFull, sizeof(double) * (lens[0] + 2) * ny2);

    if (nmb_mpi_proc > 1) {
      for (int ix = 1; ix < nmb_mpi_proc; ix++) {
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

  // endtim = MPI_Wtime();
  // tiktork = endtim - strtim;
  // if (mpi_rank == 0) {
  //   printf("send file to nodes time %f\n", tiktork);
  // }

  int bsize = 3000, ib, jb, ie, je;
  double sum = 0.0;
  // compute the temperatures in next time step
  // double itrtim = 0.0, msgtim = 0.0;
  for (int n = 0; n < niter; ++n) {
    // strtim = MPI_Wtime();
    for (ib = 1; ib < loclen + 1; ib += bsize) {
      ie = ib + bsize < loclen + 1 ? ib + bsize : loclen + 1;
      for (jb = 1; jb < ny + 1; jb += bsize) {
        je = jb + bsize < ny + 1 ? jb + bsize : ny + 1;
        for (int ix = ib; ix < ie; ++ix) {
          for (int jx = jb; jx < je; ++jx) {

            const double hijW = ivEntries[ix * (ny2) + jx - 1];
            const double hij = ivEntries[ix * (ny2) + jx];
            const double hijE = ivEntries[ix * (ny2) + jx + 1];
            const double hijN = ivEntries[(ix - 1) * (ny2) + jx];
            const double hijS = ivEntries[(ix + 1) * (ny2) + jx];

            nextTimeEntries[ix * (ny2) + jx] =
              hij * c1 + 0.25 * c0 * (hijW + hijE + hijS + hijN);
            if (n + 1 == niter) {
              sum += nextTimeEntries[ix * (ny2) + jx];
            }
          }
        }
      }
    }

    double* swp = nextTimeEntries;
    nextTimeEntries = ivEntries;
    ivEntries = swp;

    // endtim = MPI_Wtime();
    // itrtim += endtim - strtim;
    // strtim = MPI_Wtime();

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
    // endtim = MPI_Wtime();
    // msgtim += endtim - strtim;
  }

  // if (mpi_rank == 0) {
  //   printf("iteration time %f\n", itrtim);
  // }
  // if (mpi_rank == 1) {
  //   printf("msg passing time %f\n", msgtim);
  // }

  // strtim = MPI_Wtime();

  double redSum = 0.0;
  MPI_Allreduce(&sum, &redSum, 1, MPI_DOUBLE, MPI_SUM, mpiworld);
  double avg = redSum / (double)(nx * ny);

  double abssum = 0.0;
  for (ib = 1; ib < loclen + 1; ib += bsize) {
    ie = ib + bsize < loclen + 1 ? ib + bsize : loclen + 1;
    for (jb = 1; jb < ny + 1; jb += bsize) {
      je = jb + bsize < ny + 1 ? jb + bsize : ny + 1;
      for (int ix = ib; ix < ie; ++ix) {
        for (int jx = jb; jx < je; ++jx) {
          double absdelta = ivEntries[ix * (ny2) + jx] - avg;
          absdelta = (absdelta < 0.0 ? -absdelta : absdelta);
          abssum += absdelta;
        }
      }
    }
  }

  double redAbsSum = 0.0;
  MPI_Reduce(&abssum, &redAbsSum, 1, MPI_DOUBLE, MPI_SUM, 0, mpiworld);
  double absavg = redAbsSum / (double)(nx * ny);

  // endtim = MPI_Wtime();
  // tiktork = endtim - strtim;
  // if (mpi_rank == 1) {
  //   printf("compute avg time %f\n", tiktork);
  // }

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
