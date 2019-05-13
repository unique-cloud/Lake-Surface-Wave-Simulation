#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
//#define VSQR 0.1

void init(double* u, double* pebbles, int n);
// void evolve(double* un, double* uc, double* uo, double* pebbles, int n, double h, double dt, double t);
int tpdt(double* t, double dt, double end_time);
void print_heatmap(const char* filename, double* u, int n, double h);
void print_heatmap_part(const char* filename, double* u, int n, double h, int start, int height);
void init_pebbles(double* p, int pn, int n);

// void run_cpu(double* u, double* u0, double* u1, double* pebbles, int n, double h, double end_time);

extern void run_gpu(double* u, double* u0, double* u1, double* pebbles, int n, double h, double end_time, int nthreads);
extern void run_multi_gpu(double* u, double* u0, double* u1, double* pebbles, int n, int m, double h, double end_time, int nthreads, int rank, int size);

int main(int argc, char* argv[]) {
    int rank;
    int numproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);

    if (argc != 5) {
        printf("Usage: %s npoints npebs time_finish nthreads \n", argv[0]);
        return -1;
    }

    if (numproc != 4) {
        printf("This program shouble be run in exactly 4 nodes, you have %d\n", numproc);
        return -1;
    }

    int npoints = atoi(argv[1]);
    int npebs = atoi(argv[2]);
    double end_time = (double)atof(argv[3]);
    int nthreads = atoi(argv[4]);
    int narea = npoints * npoints;

    double *u_i0, *u_i1;
    double /**u_cpu, */ *u_gpu, *pebs;
    double h;

    double /*elapsed_cpu, */ elapsed_gpu;
    struct timeval /*cpu_start, cpu_end, */ gpu_start, gpu_end;

    u_i0 = (double*)malloc(sizeof(double) * narea);
    u_i1 = (double*)malloc(sizeof(double) * narea);
    pebs = (double*)malloc(sizeof(double) * narea);

    // u_cpu = (double*)malloc(sizeof(double) * narea);
    u_gpu = (double*)malloc(sizeof(double) * narea);

    if (rank == 0)
        printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

    h = (XMAX - XMIN) / npoints;

    int return_status;
    if (rank == 0)
        init_pebbles(pebs, npebs, npoints);
    return_status = MPI_Bcast((void*)pebs, narea, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (return_status != MPI_SUCCESS)
        return return_status;

    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);

    if (rank == 0)
        print_heatmap("lake_i.dat", u_i0, npoints, h);

    /*gettimeofday(&cpu_start, NULL);
    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
    gettimeofday(&cpu_end, NULL);
    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6) - (cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    printf("CPU took %f seconds\n", elapsed_cpu);*/

    gettimeofday(&gpu_start, NULL);
    // calculate the height of a block
    int height = npoints / numproc;
    double* u_i0_block = u_i0 + rank * height * npoints;
    double* u_i1_block = u_i1 + rank * height * npoints;
    double* pebs_block = pebs + rank * height * npoints;
    run_multi_gpu(u_gpu, u_i0_block, u_i1_block, pebs_block, npoints, height, h, end_time, nthreads, rank, numproc);
    gettimeofday(&gpu_end, NULL);
    elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6) - (gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
    printf("GPU took %f seconds\n", elapsed_gpu);

    char filename[15];
    sprintf(filename, "lake_f_%d.dat", rank);
    /*print_heatmap("lake_f_cpu.dat", u_cpu, npoints, h);
    print_heatmap("lake_f_gpu.dat", u_gpu, npoints, h);*/
    print_heatmap_part(filename, u_gpu, npoints, h, height * rank, height);
    free(u_i0);
    free(u_i1);
    free(pebs);
    // free(u_cpu);
    free(u_gpu);

    MPI_Finalize();

    return 0;
}

/*void run_cpu(double* u, double* u0, double* u1, double* pebbles, int n, double h, double end_time) {
    double *un, *uc, *uo;
    double t, dt;

    un = (double*)malloc(sizeof(double) * n * n);
    uc = (double*)malloc(sizeof(double) * n * n);
    uo = (double*)malloc(sizeof(double) * n * n);

    memcpy(uo, u0, sizeof(double) * n * n);
    memcpy(uc, u1, sizeof(double) * n * n);

    t = 0.;
    dt = h / 2.;

    while (1) {
        evolve(un, uc, uo, pebbles, n, h, dt, t);

        memcpy(uo, uc, sizeof(double) * n * n);
        memcpy(uc, un, sizeof(double) * n * n);

        if (!tpdt(&t, dt, end_time))
            break;
    }

    memcpy(u, un, sizeof(double) * n * n);
}*/

void init_pebbles(double* p, int pn, int n) {
    int i, j, k, idx;
    int sz;

    srand(time(NULL));
    memset(p, 0, sizeof(double) * n * n);

    for (k = 0; k < pn; k++) {
        i = rand() % (n - 4) + 2;
        j = rand() % (n - 4) + 2;
        sz = rand() % MAX_PSZ;
        idx = j + i * n;
        p[idx] = (double)sz;
    }
}

double f(double p, double t) { return -expf(-TSCALE * t) * p; }

int tpdt(double* t, double dt, double tf) {
    if ((*t) + dt > tf)
        return 0;
    (*t) = (*t) + dt;
    return 1;
}

void init(double* u, double* pebbles, int n) {
    int i, j, idx;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            idx = j + i * n;
            u[idx] = f(pebbles[idx], 0.0);
        }
    }
}

/*void evolve(double* un, double* uc, double* uo, double* pebbles, int n, double h, double dt, double t) {
    int i, j, idx;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            idx = j + i * n;

            if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                un[idx] = 0.;
            } else {
                // un[idx] = 2 * uc[idx] - uo[idx] + VSQR * (dt * dt) * ((uc[idx - 1] + uc[idx + 1] + uc[idx + n] + uc[idx - n] - 4 * uc[idx]) / (h * h) + f(pebbles[idx], t));
                un[idx] = 2 * uc[idx] - uo[idx] + VSQR * (dt * dt) * ((uc[idx - 1] + uc[idx + 1] + uc[idx + n] + uc[idx - n] + 0.25 * (uc[idx - 1 - n] + uc[idx - 1 + n] + uc[idx + 1 - n] + uc[idx + 1 + n]) - 5 * uc[idx]) / (h * h) + f(pebbles[idx], t));
            }
        }
    }
}*/

void print_heatmap(const char* filename, double* u, int n, double h) {
    int i, j, idx;

    FILE* fp = fopen(filename, "w");

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            idx = j + i * n;
            fprintf(fp, "%f %f %f\n", i * h, j * h, u[idx]);
        }
    }

    fclose(fp);
}

void print_heatmap_part(const char* filename, double* u, int n, double h, int start, int height) {
    int i, j, idx;

    FILE* fp = fopen(filename, "w");

    for (i = start; i < start + height; i++) {
        for (j = 0; j < n; j++) {
            idx = j + i * n;
            fprintf(fp, "%f %f %f\n", i * h, j * h, u[idx - start * n]);
        }
    }

    fclose(fp);
}
