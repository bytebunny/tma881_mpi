[//]: # (To preview markdown file in Emacs type C-c C-c p)

## My MPI implementation of the assignment

The idea is to distribute the work between processes by splitting the temperature grid row-wise, so that each process works on its number of rows. The communication is then done after each time increment only for the interfacing rows, i.e. rows that belong to both processes. This way the communication is reduced to minimum.

Reading of the input is handled by the master process. The input parameters (conductivity coefficient, number of time increments, as well as width and height of the grid) are stored in a structure and distributed to the rest of the processes using `MPI_Bcast()`. However, sending of a custom structure whose set of data (a double and three integers) is not covered by one of the MPI's standard data types (e.g. `MPI_DOUBLE_INT`, that is a double and an integer) requires creation of a custom `MPI_Datatype`. This was done by means of `MPI_Type_create_struct()`. Note that in order for MPI to be able to recognise the new data type a call to `MPI_Type_commit()` must be made giving the address to the new data type.

The computations on the grid are split between the processes row-wise, i.e. each process (that has rank < total number of rows) receives a number of rows to process. The communication is carried out only for the top and bottom rows that the neighbouring processes depend on. The first (master) and the last processes communicate only their bottom and
top row, respectively.

After the computations are completed, the averages were computed by means of `MPI_Allreduce()` and `MPI_Reduce()`, where the former one distrubutes the result across all the
processes in a group, while the latter sends the result only to root (master process in this case).

/Rostyslav
