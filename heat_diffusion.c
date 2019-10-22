#include "heat_diffusion.h"

int main(int argc, char* argv[])
{
  int niter=0;
  double c=0.;
  char* ptr1 = NULL;
  char* ptr2 = NULL;
  if (argc == 3 ) {
    for ( size_t ix = 1; ix < argc; ++ix ) {
      ptr1 = strchr(argv[ix], 'n' );
      ptr2 = strchr(argv[ix], 'd' );
      if ( ptr1 ){
  	niter = strtol(++ptr1, NULL, 10);
      } else if ( ptr2 ) {
  	c = atof(++ptr2);
      }
    }
  } else {
    printf("Invalid number of arguments. Correct syntax is: heat_diffusion -n#numberOfTimeSteps4 -d#diffusionConstant\n");
    exit(0);
  }

  int nx=0, ny=0;
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
  sscanf(line, "%d %d", &nx, &ny);
  //store initial values (note: row major order)
  double* ivEntries = (double*) malloc( sizeof(double)*nx*ny);
  for (size_t ix = 0; ix < nx*ny; ++ix) {
    ivEntries[ix] = 0;
  }
  //read the rest
  while ( fgets(line, sizeof(line), input) != NULL) {
    sscanf(line, "%d %d %lf", &i, &j, &t);
    ivEntries[i*ny + j] = t;
  }
  fclose(input);

  double* nextTimeEntries = (double*) malloc(sizeof(double)*nx*ny);

  //compute the temperatures in next time step
  for ( int n = 0; n < niter; ++n) {
    for ( int ix = 0; ix < nx; ++ix){
      for ( int jx = 0; jx < ny; ++jx) {
        double hij, hijW, hijE, hijS, hijN;
        hij = ivEntries[ix*ny + jx];
        hijW = ( jx-1 >= 0 ? ivEntries[ix*ny + jx-1] : 0. );
        hijE = ( jx+1 < ny ? ivEntries[ix*ny + jx+1] : 0.);
        hijS = ( ix+1 < nx ? ivEntries[(ix+1)*ny + jx] : 0.);
        hijN = ( ix-1 >= 0 ? ivEntries[(ix-1)*ny + jx] : 0.);
        
        nextTimeEntries[ix*ny + jx] = computeNextTemp(&hij, &hijW, &hijE, &hijS, &hijN, &c);
      }
    }
    memcpy(ivEntries, nextTimeEntries, nx*ny*sizeof(double));
  }

  //compute average temperature
  double avg = computeAverage(nextTimeEntries, nx, ny);

  double* avgDiffEntries = (double*) malloc(sizeof(double)*nx*ny);
  for (int ix = 0; ix < nx*ny; ++ix) {
    avgDiffEntries[ix] = 0.;
  }

  /* //compute differences (in nextTimeEntries since the contents are copied to ivEntries) */
  if ( niter != 0) {
    memcpy(avgDiffEntries, nextTimeEntries, nx*ny*sizeof(double));
  } else {
    memcpy(avgDiffEntries, ivEntries, nx*ny*sizeof(double));
  }
  for ( int ix = 0; ix < nx*ny; ++ix ) {
    avgDiffEntries[ix] -= avg;
    avgDiffEntries[ix] = ( avgDiffEntries[ix] < 0 ? avgDiffEntries[ix]*-1.: avgDiffEntries[ix] );
  }

  double avgDiff = computeAverage(avgDiffEntries, nx, ny);
  printf("average: %e\n", avg);
  printf("average absolute difference: %e\n", avgDiff);

  free(avgDiffEntries);
  free(nextTimeEntries);
  free(ivEntries);
  
  return 0;
}

double computeNextTemp(const double* hij, const double* hijW, const double* hijE, const double* hijS, const double* hijN, const double* c)
{
  return (*hij) + (*c)*( ((*hijW) + (*hijE) + (*hijS) + (*hijN))/4 - (*hij));
}

double computeAverage(const double* tempArray, int nx, int ny)
{
  double sum=0.;
  for ( int ix = 0; ix < nx*ny; ++ix ) {
    sum += tempArray[ix];
  }
  return sum/(nx*ny);
}
