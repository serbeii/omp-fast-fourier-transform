#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FFT_TASK_CUTOFF 4096
#define FFT_LOOP_CUTOFF 8192
//#define DEBUG

typedef complex double cplx;
int dynamic_scheduling = 0;

void parallel_fft(cplx x1[], cplx x2[], int N);
void fft(cplx x[], int N, int s);
void generate_data(cplx* x, int N);

int main(int argc, char** argv) {
    int signal_size = 4096;
    if (argc > 1) {
        signal_size = atoi(argv[1]);
        if (argc == 3) {
            omp_set_num_threads(atoi(argv[2]));
        }
    } else {
        omp_set_num_threads(omp_get_max_threads());
    }

    printf("1\n");
    cplx* data = (cplx*)malloc(signal_size * sizeof(cplx));
    printf("2\n");
    generate_data(data, signal_size);
    cplx* data_clone = (cplx*)malloc(signal_size * sizeof(cplx));
    data_clone = memcpy(data_clone, data, signal_size * sizeof(cplx));
    printf("3\n");

#ifdef DEBUG
    printf("Original data: ");
    for (int i = 0; i < signal_size; i++) {
        if (cimag(data[i]) == 0)
            printf("%.2f ", creal(data[i]));
        else
            printf("%.2f%+.2fi ", creal(data[i]), cimag(data[i]));
    }
    printf("\n");
#endif
    printf("Starting\n");
    parallel_fft(data, data_clone, signal_size);

#ifdef DEBUG
    printf("Static FFT result: ");
    for (int i = 0; i < signal_size; i++) {
        printf("%.2f%+.2fi ", creal(data[i]), cimag(data[i]));
    }
    printf("\n");
    printf("Dynamic FFT result: ");
    for (int i = 0; i < signal_size; i++) {
        printf("%.2f%+.2fi ", creal(data_clone[i]), cimag(data_clone[i]));
    }
    printf("\n");
#endif
    printf("Done\n");
    return 0;
}

void parallel_fft(cplx x1[], cplx x2[], int N) {
    double start, finish, elapsed;
    start = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        fft(x1, N, 1);
    }
    finish = omp_get_wtime();
    elapsed = finish - start;
    printf("Guided scheduling finished in %e seconds.\n", elapsed);

    start = omp_get_wtime();
    dynamic_scheduling = 1;
#pragma omp parallel
    {
#pragma omp single
        fft(x2, N, 1);
    }
    finish = omp_get_wtime();
    elapsed = finish - start;
    printf("Dynamic scheduling finished in %e seconds.\n", elapsed);
}

void fft(cplx x[], int N, int s) {
    if (N == 1) {
        return;
    } else {
        if (N > FFT_TASK_CUTOFF) {
#pragma omp task
            fft(x, N / 2, 2 * s);
#pragma omp task
            fft(x + s, N / 2, 2 * s);
#pragma omp taskwait
        } else {
            fft(x, N / 2, 2 * s);
            fft(x + s, N / 2, 2 * s);
        }
        if (N / 2 > FFT_LOOP_CUTOFF) {
            if (dynamic_scheduling == 1) {
#pragma omp for schedule(dynamic) nowait
                for (int k = 0; k < N / 2; k++) {
                    cplx p = x[k * 2 * s];
                    cplx q = cexp(-2 * M_PI * I * k / N) * x[k * 2 * s + s];
                    x[k * 2 * s] = p + q;
                    x[k * 2 * s + s] = p - q;
                }
            } else {
#pragma omp for schedule(guided) nowait
                for (int k = 0; k < N / 2; k++) {
                    cplx p = x[k * 2 * s];
                    cplx q = cexp(-2 * M_PI * I * k / N) * x[k * 2 * s + s];
                    x[k * 2 * s] = p + q;
                    x[k * 2 * s + s] = p - q;
                }
            }
        } else {
            for (int k = 0; k < N / 2; k++) {
                cplx p = x[k * 2 * s];
                cplx q = cexp(-2 * M_PI * I * k / N) * x[k * 2 * s + s];
                x[k * 2 * s] = p + q;
                x[k * 2 * s + s] = p - q;
            }
        }
    }
}

void generate_data(cplx* x, int N) {
    srand(time(NULL));
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = rand();
    }
}
