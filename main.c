#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int thread_count = 1;

#define FFT_TASK_CUTOFF 4096
// #define DEBUG

typedef complex double cplx;

// Scheduling types
typedef enum {
    SCHED_DEFAULT, // Default task-based scheduling
    SCHED_DYNAMIC  // Dynamic scheduling
} scheduling_type_t;

void parallel_fft(cplx x[], int N, scheduling_type_t sched_type, int chunk_size);
void fft(cplx x[], int N, int s);
void generate_data(cplx* x, int N);

int main(int argc, char** argv) {
    int signal_size = 4096;
    scheduling_type_t sched_type = SCHED_DEFAULT; // Default Scheduling
    int chunk_size = 16; // Default chunk size

    if (argc > 1) {
        signal_size = atoi(argv[1]);
        if (argc > 2) {
            thread_count = atoi(argv[2]);
            if (argc > 3) {
                // Parse scheduling type
                if (strcmp(argv[3], "default") == 0) {
                    sched_type = SCHED_DEFAULT;
                } else if (strcmp(argv[3], "dynamic") == 0) {
                    sched_type = SCHED_DYNAMIC;
                }
                
                if (argc > 4) {
                    chunk_size = atoi(argv[4]);
                }
            }
        }
    } else {
        thread_count = omp_get_max_threads();
    }

    const char* sched_names[] = {"default", "dynamic"};

    printf("Running FFT with parameters:\n");
    printf("Signal size: %d\n", signal_size);
    printf("Threads: %d\n", thread_count);
    printf("Scheduling: %s\n", sched_names[sched_type]);
    printf("Chunk size: %d\n", chunk_size);
    
    cplx* data = (cplx*)malloc(signal_size * sizeof(cplx));

    generate_data(data, signal_size);

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
    parallel_fft(data, signal_size, sched_type, chunk_size);

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

void parallel_fft(cplx x[], int N, scheduling_type_t sched_type, int chunk_size) {
    omp_set_num_threads(thread_count);
    
    if (sched_type == SCHED_DEFAULT) {
        // Original task-based implementation
        #pragma omp parallel
        {
            #pragma omp single
            {
                printf("Running with default task-based scheduling\n");
                fft(x, N, 1);
            }
        }
    }
    else if(sched_type == SCHED_DYNAMIC) {
        printf("Running with dynamic scheduling, chunk size = %d\n", chunk_size);
        
        int stages = 0;
        int temp = N;
        while (temp > 1) {
            temp >>= 1;
            stages++;
        }
        
        int m = 1;
        for (int stage = 0; stage < stages; stage++) {
            int m2 = m * 2;
            cplx wm = cexp(-2.0 * M_PI * I / m2);
            
            // Process each group of butterflies in parallel with dynamic scheduling
            #pragma omp parallel for schedule(dynamic, chunk_size)
            for (int k = 0; k < N; k += m2) {
                cplx w = 1.0;
                for (int j = 0; j < m; j++) {
                    cplx t = w * x[k + j + m];
                    cplx u = x[k + j];
                    x[k + j] = u + t;
                    x[k + j + m] = u - t;
                    w *= wm;
                }
            }
            m = m2;
        }
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