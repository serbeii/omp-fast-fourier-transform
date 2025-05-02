#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int thread_count = 1;

#define FFT_TASK_CUTOFF 4096
// #define DEBUG

typedef complex double cplx;

void parallel_fft(cplx x[], int N);
void fft(cplx x[], int N, int s);
void generate_data(cplx* x, int N);

int main(int argc, char** argv) {
    int signal_size = 4096;
    if (argc > 1) {
        signal_size = atoi(argv[1]);
        if (argc == 3) {
            thread_count = atoi(argv[2]);
        }
    } else {
        thread_count = omp_get_max_threads();
    }

    printf("1\n");
    cplx* data = (cplx*)malloc(signal_size * sizeof(cplx));
    printf("2\n");
    generate_data(data, signal_size);
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
    parallel_fft(data, signal_size);

#ifdef DEBUG
    printf("FFT result: ");
    for (int i = 0; i < signal_size; i++) {
        printf("%.2f%+.2fi ", creal(data[i]), cimag(data[i]));
    }
    printf("\n");
#endif
    printf("Done\n");
    return 0;
}

void parallel_fft(cplx x[], int N) {
#pragma omp parallel
    {
#pragma omp single
        fft(x, N, 1);
    }
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
        for (int k = 0; k < N / 2; k++) {
            cplx p = x[k * 2 * s];
            cplx q = cexp(-2 * M_PI * I * k / N) * x[k * 2 * s + s];
            x[k * 2 * s] = p + q;
            x[k * 2 * s + s] = p - q;
        }
    }
}

void generate_data(cplx* x, int N) {
    srand(time(NULL));
#pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < N; i++) {
        x[i] = rand();
    }
}
