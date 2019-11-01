#include "heat_diffusion.h"
#include <mpi.h>
#include <stddef.h> // offsetof() macro.

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int nmb_mpi_proc, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nmb_mpi_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // Declare variables for all processes (to skip declarations in every scope):
  double conductivity;
  int niter, width, height, total_size;
  double hij, hijW, hijE, hijS, hijN;
  struct input_struct{
    double c;
    int    n, w, h;
  } inputs;
  double * grid_init; // pointer to memory with initial temperatures.

 if (mpi_rank == 0) /////////////////////// master process /////////////////////
   {
      char* ptr1 = NULL;
      char* ptr2 = NULL;
      if (argc == 3 ) {
        for ( int ix = 1; ix < argc; ++ix ) {
          ptr1 = strchr(argv[ix], 'n' );
          ptr2 = strchr(argv[ix], 'd' );
          if ( ptr1 ){
            niter = strtol(++ptr1, NULL, 10);
          } else if ( ptr2 ) {
            conductivity = atof(++ptr2);
          }
        }
      } else {
        printf("Invalid number of arguments. Correct syntax is: heat_diffusion -n#numberOfTimeSteps4 -d#diffusionConstant\n");
        exit(0);
      }
      
      char line[80];
      FILE *input = fopen("diffusion", "r");
      int i=0, j=0;
      double t=0.;
      if ( input == NULL ) {
        perror("Error opening file");
        exit(0);
      }
      //read the first line
      fgets( line, sizeof(line), input);
      sscanf(line, "%d %d", &width, &height);
      total_size = width * height;
      //store initial values (note: row major order)
      grid_init = (double*) malloc( sizeof(double) * total_size);
      for (int ix = 0; ix < total_size; ++ix) {
        grid_init[ix] = 0;
      }
      //read the rest
      while ( fgets(line, sizeof(line), input) != NULL) {
        sscanf(line, "%d %d %lf", &j, &i, &t);
        grid_init[i*width + j] = t;
      }
      fclose(input);

      inputs.c = conductivity;
      inputs.n = niter;
      inputs.w = width;
      inputs.h = height;
   }
 //////////////////////////// all processes ////////////////////////////////////
 // Create custom MPI data type for the structure inputs:
 MPI_Datatype mpi_double_3int;
 int block_lengths[4] = {1,1,1,1}; // 1 double and 3 single integers in structure. 
 MPI_Aint offsets[4];
 offsets[0] = offsetof( struct input_struct, c );
 offsets[1] = offsetof( struct input_struct, n );
 offsets[2] = offsetof( struct input_struct, w );
 offsets[3] = offsetof( struct input_struct, h );
 MPI_Datatype types[4] = {MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT};
 MPI_Type_create_struct( 4, // number of elements in input structure.
                         block_lengths, offsets, types,
                         &mpi_double_3int);
 MPI_Type_commit(&mpi_double_3int);
 
 // Send inputs to all processes: 
 // All the processes have to execute this statement.
 MPI_Bcast( &inputs, 1, // 1 structure.
            mpi_double_3int,
            0, // master sends out inputs.
            MPI_COMM_WORLD );

 if (mpi_rank == 0) ////////////////// master process //////////////////////////
   {
     // distribute the work (each process recieves only its part of array):
     int row_start = 0, row_end = 0; // initialize to prevent execution when rank > row number
     for ( int mpi_proc = 1; // master (0) already has access to its part.
           mpi_proc < nmb_mpi_proc;
           ++mpi_proc )
       {
         if ( nmb_mpi_proc <= height) {
           // to process row i rows i-1 and i+1 are needed:
           row_start = mpi_proc * height / nmb_mpi_proc - 1; 
           row_end = (mpi_proc + 1)* height / nmb_mpi_proc;
         }
         else{ // if there are more processes than rows:
           if ( mpi_proc < height ){
             row_start = mpi_proc - 1;
             row_end = mpi_proc + 1;
           }
         }
         // All worker processes (that will work) but last need one more row:
         if ( ( mpi_proc + 1 < nmb_mpi_proc ) &&
              ( mpi_proc + 1 < height ) ) row_end += 1;
         
         if (row_end - row_start > 0){ // if there is work for this process:
           MPI_Send( grid_init + row_start * width, // address of send buffer.
                     (row_end - row_start) * width, // number of elements.
                     MPI_DOUBLE, mpi_proc, // destination.
                     mpi_proc, MPI_COMM_WORLD );
         }
       }
     
     //////////////////////////// Update temperatures //////////////////////////
     // Decide on the rows of the grid handled by the master process:
     row_start = 0; // master starts from the 1st row.
     if ( nmb_mpi_proc <= height) row_end = height / nmb_mpi_proc;
     else row_end = 1; // if there are more processes than rows.

     // Allocate space for rows to be updated and 1 interface row to receive 
     double * grid_local_new = (double *) malloc( ( row_end + 1 - row_start ) *
                                                  width * sizeof(double) );
     double * temp_ptr; // point to swap old and new temperatures.

     for ( int iter = 0; iter < niter; ++iter ) {
       for ( int ix = row_start; ix < row_end; ++ix ) { // row start is always 0, so there is no shift (unlike worker).
         for ( int jx = 0; jx < width; ++jx )
           {
             hij = grid_init[ ix * width +jx ];
             hijW = ( jx-1 >= 0 ? grid_init[ ix * width + jx-1 ] : 0. );
             hijE = ( jx+1 < width ? grid_init[ ix * width + jx+1 ] : 0.);
             hijS = ( ix+1 < height ? grid_init[ (ix+1) * width + jx ] : 0.);
             hijN = ( ix-1 >= 0 ? grid_init[ (ix-1) * width + jx ] : 0.);
             
             grid_local_new[ ix * width + jx] = (1. - conductivity) * hij +
               conductivity * 0.25 * ( hijW + hijE + hijS + hijN );
           }
       }
      
       if ( nmb_mpi_proc > 1 ) {
         MPI_Status status;
         // Update the boundary shared with the next process:
         MPI_Sendrecv( grid_local_new + (row_end - row_start - 1) * width, // master sends its last updated row.
                       width, MPI_DOUBLE,
                       mpi_rank + 1, // destination.
                       mpi_rank, // send tag
                       // receive:
                       grid_local_new + (row_end - row_start) * width, // move pointer to bottom row.
                       width, MPI_DOUBLE,
                       mpi_rank + 1, // the next worker has local bottom row.
                       mpi_rank + 1, // receive tag
                       MPI_COMM_WORLD, &status );
       }

       if ( iter == niter - 1 ) break; // don't swap at the last step
       // Swap new and old arrays:
       temp_ptr = grid_init;
       grid_init = grid_local_new;
       grid_local_new = temp_ptr;
     }

     ////////////////////////// Compute average temperature ////////////////////
     double sum0 = 0, sum1 = 0, sum2 = 0;
     for ( int ix = row_start; ix < row_end; ++ix ) { // row start is always 0, so there is no shift (unlike worker).
       for ( int jx = 0; jx < width; jx += 3 )
           {
             sum0 += grid_local_new[ ix * width + jx     ];
             sum1 += grid_local_new[ ix * width + jx + 1 ];
             sum2 += grid_local_new[ ix * width + jx + 2 ];
           }
     }
     double sum_master = sum0 + sum1 + sum2;
     
     double sum_total;
     MPI_Allreduce( &sum_master, // send buffer
                    &sum_total, 1, // receive one sent element
                    MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD );
     double avg = sum_total / total_size;
     printf("average: %e\n", avg);

     //////////////////////// Compute average of abs diff //////////////////////
     sum0 = 0, sum1 = 0, sum2 = 0;
     for ( int ix = row_start; ix < row_end; ++ix ) { // row start is always 0, so there is no shift (unlike worker).
       for ( int jx = 0; jx < width; jx += 3 )
           {
             grid_local_new[ ix * width + jx ] -= avg;
             sum0 += ( grid_local_new[ ix * width + jx ] < 0. ?
                       -1. * grid_local_new[ ix * width + jx ] :
                       grid_local_new[ ix * width + jx ] );

             grid_local_new[ ix * width + jx + 1] -= avg;
             sum1 += ( grid_local_new[ ix * width + jx + 1 ] < 0. ?
                       -1. * grid_local_new[ ix * width + jx + 1 ] :
                       grid_local_new[ ix * width + jx + 1 ] );

             grid_local_new[ ix * width + jx + 2] -= avg;
             sum2 += ( grid_local_new[ ix * width + jx + 2 ] < 0. ?
                       -1. * grid_local_new[ ix * width + jx + 2 ] :
                       grid_local_new[ ix * width + jx + 2 ] );
           }
     }
     double diff_sum_master = sum0 + sum1 + sum2;
     // Reduce to master only:
     MPI_Reduce( &diff_sum_master, // send buffer
                 &sum_total, 1, // receive one sent element
                 MPI_DOUBLE, MPI_SUM,
                 mpi_rank, // root
                 MPI_COMM_WORLD );
     double diff_avg = sum_total / total_size;
     printf("average of abs diff: %e\n", diff_avg);

     free(grid_local_new);
     free(grid_init);
   }
 else ////////////////////////// worker processes //////////////////////////////
   {
     // receive the part of array to work on:
     conductivity = inputs.c;
     niter = inputs.n;
     width = inputs.w;
     height = inputs.h;
     total_size = width * height;

     // Decide on the rows of the grid handled by the worker process:
     int row_start = 0, row_end = 0; // initialize to prevent execution when rank > row number
     if ( nmb_mpi_proc <= height) {
       row_start = mpi_rank * height / nmb_mpi_proc - 1; 
       row_end =  (mpi_rank + 1) * height / nmb_mpi_proc;
     }
     else{ // if there are more processes than rows:
       if ( mpi_rank < height ){
         row_start = mpi_rank - 1;
         row_end = mpi_rank + 1;
       }
     }

     int n_rows_worker = row_end - row_start;
     double * grid_local = (double *) malloc( (n_rows_worker + 1) * // the last worker does not need the extra row, but allocate it anyway.
                                              width * sizeof(double) );
     MPI_Status status;
     if ( n_rows_worker > 0){ // if there is work for this process:
       MPI_Recv( grid_local, // receiving buffer
                 (n_rows_worker+1) * width, // number of elements
                 MPI_DOUBLE, 0, // sending process
                 mpi_rank, MPI_COMM_WORLD, &status);
     }
     //////////////////////////// Update temperatures //////////////////////////
     double * grid_local_new = (double *) malloc( (n_rows_worker + 1) *  // the last worker does not need the extra row, but allocate it anyway.
                                                  width * sizeof(double) );
     double * temp_ptr; // point to swap old and new temperatures.

     for ( int iter = 0; iter < niter; ++iter ) {
       for ( int ix = 1; // the 1st row is read-only for worker processes
             ix < n_rows_worker; ++ix ) {
         for ( int jx = 0; jx < width; ++jx )
           {
             hij = grid_local[ ix * width +jx ];
             hijW = ( jx-1 >= 0 ? grid_local[ ix * width + jx-1 ] : 0. );
             hijE = ( jx+1 < width ? grid_local[ ix * width + jx+1 ] : 0.);
             hijS = ( row_start + ix + 1 < height ? grid_local[ (ix+1) * width + jx ] : 0.);
             hijN = grid_local[ (ix-1) * width + jx ]; // worker can never process the 1st row, so conditional is not necessary.
             
             grid_local_new[ ix * width + jx] = (1. - conductivity) * hij +
               conductivity * 0.25 * ( hijW + hijE + hijS + hijN );
           }
       }
       
       if ( n_rows_worker > 0){ // if there is work for this process:
         // Update the boundary shared with the previous process:
         MPI_Sendrecv( grid_local_new + width, // worker sends its 1st updated row.
                       width, MPI_DOUBLE,
                       mpi_rank - 1, // 1st updated row goes to previous worker.
                       mpi_rank, // send tag
                       // receive:
                       grid_local_new, // receive the local top row from previous process.
                       width, MPI_DOUBLE,
                       mpi_rank - 1, // source: previous process has local top row.
                       mpi_rank - 1, // receive tag
                       MPI_COMM_WORLD, &status );
         
         if ( ( mpi_rank + 1 < nmb_mpi_proc ) && // if there is a next process:
              ( mpi_rank + 1 < height ) ) { // and the next process should work:
           // Update the boundary shared with the next process:
           MPI_Sendrecv( grid_local_new + (n_rows_worker - 1) * width, // worker sends its last updated row.
                         width, MPI_DOUBLE,
                         mpi_rank + 1, // last updated row goes to next worker.
                         mpi_rank, // send tag
                         // receive:
                         grid_local_new  + (n_rows_worker - 0) * width, // move pointer to bottom row.
                         width, MPI_DOUBLE,
                         mpi_rank + 1, // source: next process has local bottom row.
                         mpi_rank + 1, // receive tag
                         MPI_COMM_WORLD, &status ); 
         }
       }
       
       if ( iter == niter - 1 ) break; // don't swap at the last step
       // Swap new and old arrays:
       temp_ptr = grid_local;
       grid_local = grid_local_new;
       grid_local_new = temp_ptr;
     }

     ////////////////////////// Compute average temperature ////////////////////
     double sum0 = 0, sum1 = 0, sum2 = 0;
     for ( int ix = 1; // the 1st row is read-only for worker processes
           ix < n_rows_worker; ++ix ) {
       for ( int jx = 0; jx < width; jx += 3 )
           {
             sum0 += grid_local_new[ ix * width + jx     ];
             sum1 += grid_local_new[ ix * width + jx + 1 ];
             sum2 += grid_local_new[ ix * width + jx + 2 ];
           }
     }
     double sum_worker = sum0 + sum1 + sum2;

     double sum_total;
     MPI_Allreduce( &sum_worker, // send buffer
                    &sum_total, 1, // receive one sent element
                    MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD );
     double avg = sum_total / total_size;

     //////////////////////// Compute average of abs diff //////////////////////
     sum0 = 0, sum1 = 0, sum2 = 0;
     for ( int ix = 1; // the 1st row is read-only for worker processes
           ix < n_rows_worker; ++ix ) {
       for ( int jx = 0; jx < width; jx += 3 )
           {
             grid_local_new[ ix * width + jx ] -= avg;
             sum0 += ( grid_local_new[ ix * width + jx ] < 0. ?
                       -1. * grid_local_new[ ix * width + jx ] :
                       grid_local_new[ ix * width + jx ] );

             grid_local_new[ ix * width + jx + 1] -= avg;
             sum1 += ( grid_local_new[ ix * width + jx + 1 ] < 0. ?
                       -1. * grid_local_new[ ix * width + jx + 1 ] :
                       grid_local_new[ ix * width + jx + 1 ] );

             grid_local_new[ ix * width + jx + 2] -= avg;
             sum2 += ( grid_local_new[ ix * width + jx + 2 ] < 0. ?
                       -1. * grid_local_new[ ix * width + jx + 2 ] :
                       grid_local_new[ ix * width + jx + 2 ] );
           }
     }
     double diff_sum_worker = sum0 + sum1 + sum2;

     // Reduce to master only:
     MPI_Reduce( &diff_sum_worker, // send buffer
                 &sum_total, 1, // receive one sent element, meaningfull only at root
                 MPI_DOUBLE, MPI_SUM,
                 0, // root
                 MPI_COMM_WORLD );

     free(grid_local_new);
     free(grid_local);
   }
 
 MPI_Finalize();
 
 return 0;
}
