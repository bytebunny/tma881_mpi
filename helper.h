#include "omp.h"
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef _H_HELPER // once-only header
#define _H_HELPER

void
computeNextTemp(size_t ,
                size_t ,
                const double* ,
                int ,
                int ,
                double ,
                double* );
double
computeAverage(const double* , int , int );
;

#endif
