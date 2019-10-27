#ifndef HEATDIFF_H
#define HEATDIFF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double computeNextTemp(const double* hij, const double* hijW, const double* hijE, const double* hijS, const double* hijN, const double* c);
double computeAverage(const double* tempArray, int nx, int ny);

#endif
