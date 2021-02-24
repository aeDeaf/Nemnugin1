#define NUM_THREADS 12

#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
const double eps = 0.001;
const int N = 30000;

double** A;
double** B;
double BNorm;
double* rowSums;
double* F;
double* X;


int signum(int num) {
	if (num >= 0) 
	{
		return 1;
	}
	else {
		return -1;
	}
}

void Jacobi(int N, double** A, double* F, double* X, double q)
{
	double* TempX = new double[N];
	double* TempX2 = new double[N];
	double norm;
	double realEps = (1 - q) * eps / q;

		do {
#pragma omp parallel 
			{
#pragma omp for
				for (int i = 0; i < N; i++) {
					TempX[i] = F[i];
					for (int g = 0; g < N; g++) {
						if (i != g)
							TempX[i] -= A[i][g] * X[g];
					}
					TempX[i] /= A[i][i];
				}
			}
			norm = 0;
#pragma omp parallel reduction(+:norm)
			{
#pragma omp for
				for (int i = 0; i < N; i++) {
					norm += pow(TempX[i] - X[i], 2);
				}
#pragma omp for
				for (int h = 0; h < N; h++) {
					X[h] = TempX[h];
				}

			}

		} while (norm > realEps);
	
	printf("%f\n", norm);
	delete[] TempX;
	delete[] TempX2;
}

void run(int numThreads) {
	omp_set_num_threads(numThreads);
	for (int i = 0; i < N; i++) {
		X[i] = F[i];
	}
	double tim = omp_get_wtime();

	Jacobi(N, A, F, X, BNorm);
	double workTime = omp_get_wtime() - tim;
	printf("Num threads: %d\n", numThreads);
	printf("Execution time: %f\n", workTime);
}

int main() {

	double tim;




	rowSums = new double[N];
	A = new double* [N];
	F = new double[N];
	X = new double[N];
	for (int i = 0; i < N; i++) {
		A[i] = new double[N];
	}
	for (int i = 0; i < N; i++) {
		rowSums[i] = 0;
		for (int j = 0; j < N; j++) {
			A[i][j] = (double) rand() / (RAND_MAX + 1);
		}
	}

	for (int i = 0; i < N; i++) {
		A[i][i] = N;
	}

	
	for (int i = 0; i < N; i++) {
		F[i] = (double) rand() / RAND_MAX;
		X[i] = F[i];
	}
	/*for (int i = 0; i < N; i++) {
		F[i] = 1;
		X[i] = 1;
		for (int j = 0; j < N; j++) {
			if (i == j) {
				A[i][j] = 5;
			}
			else {
				A[i][j] = 1;
			}
		}
	}*/
	BNorm = 0;
	for (int i = 0; i < N; i++) {
		double curSum = 0;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				curSum += fabs(-A[i][j] / A[i][i]);
			}
			if (curSum > BNorm) {
				BNorm = curSum;
			}
		}
	}
	printf("%f\n", BNorm);
	printf("Start run\n");
	for (int i = 1; i <= 12; i++) {
		run(i);
	}
	/*for (int i = 0; i < N; i++) {
		printf("%f\n", X[i]);
	}*/
	return 0;
}

