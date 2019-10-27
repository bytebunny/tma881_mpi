#include <stdio.h>
#include <mpi.h>

int main(
    int argc,
    char * argv[]
    )
{
  int a = 3; // still a private variable for each process. This is a bas programming style, this statement should be moved after MPI_Init().
  MPI_Init(&argc, &argv);

  int nmb_mpi_proc, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nmb_mpi_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  a += 1;
  printf("%d\n", a);

  if (mpi_rank==0) { // master
    printf( "Number of processes: %d\n", nmb_mpi_proc );
    { // grouping creates scope for variables like msg.
      int msg = 1; int len = 1; int dest_rank = 1; int tag = 1;
      MPI_Send(&msg, len, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
      printf( "MPI message sent from %d: %d\n", mpi_rank, msg );
    }

    {
      int msg; int max_len = 1; int src_rank = 1; int tag = 1;
      MPI_Status status;
      MPI_Recv(&msg, max_len, MPI_INT, src_rank, tag, MPI_COMM_WORLD, &status);
      printf( "MPI message received at %d: %d\n", mpi_rank, msg );
    }
  }

  else if (mpi_rank==1) { // worker
    int msg;

    {
      int max_len = 1; int src_rank = 0; int tag = 1;
      MPI_Status status;
      MPI_Recv(&msg, max_len, MPI_INT, src_rank, tag, MPI_COMM_WORLD, &status);
      printf( "MPI message received at %d: %d\n", mpi_rank, msg );
    }
    
    ++msg;
    
    {
      int len = 1; int dest_rank = 0; int tag = 1;
      MPI_Send(&msg, len, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
      printf( "MPI message sent from %d: %d\n", mpi_rank, msg );
    }
  }
  
  MPI_Finalize();

  return 0;
}
