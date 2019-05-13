#include "jemalloc/jemalloc.h"
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
#define VSQR 0.1

#define NPOINTS 128
#define NAREA (NPOINTS * NPOINTS)
#define NPEBS 8
#define END_TIME 1.0

#define BACK_FILE "app.back"
#define MMAP_FILE "app.mmap"
#define MMAP_SIZE ((size_t)1 << 30)
#define NITERATIONS 20

void init(double* u, double* pebbles, int n);
void evolve(double* un, double* uc, double* uo, double* pebbles, int n, double h, double dt, double t);
int tpdt(double* t, double dt, double end_time);
void print_heatmap(const char* filename, double* u, int n, double h);
void init_pebbles(double* p, int pn, int n);

void run_cpu(double* u, double* u0, double* u1, double* pebbles, int n, double h, double end_time);

PERM double u_i0[NAREA];
PERM double u_i1[NAREA];
PERM double pebs[NAREA];
PERM double t;
PERM double h;
PERM double dt;
PERM struct timeval cpu_start;
PERM int iteration_size;
PERM int checkpoint;

int main(int argc, char* argv[]) {
    int do_restore = argc > 1 && strcmp("-r", argv[1]) == 0;
    const char* mode = (do_restore) ? "r+" : "w+";

    perm(PERM_START, PERM_SIZE);
    mopen(MMAP_FILE, mode, MMAP_SIZE);
    bopen(BACK_FILE, mode);

    int npoints = NPOINTS;
    int npebs = NPEBS;
    double end_time = END_TIME;
    int narea = NAREA;

    double elapsed_cpu;
    struct timeval cpu_end;

    double* u_cpu = (double*)malloc(sizeof(double) * narea);

    if (!do_restore) {
        t = 0.;
        h = (XMAX - XMIN) / npoints;
        dt = h / 2.;
        iteration_size = ceil(end_time / dt) / NITERATIONS;
        iteration_size = iteration_size > 0 ? iteration_size : 1;
        checkpoint = 0;
        init_pebbles(&pebs, npebs, npoints);
        init(&u_i0, &pebs, npoints);
        init(&u_i1, &pebs, npoints);
        print_heatmap("lake_i.dat", &u_i0, npoints, h);
        gettimeofday(&cpu_start, NULL);
        mflush();
        backup();
    } else {
        if (checkpoint > 0)
            printf("restarting from checkpoint %d\n", checkpoint);
        else
            printf("restarting from beginning\n");

        restore();
    }

    run_cpu(u_cpu, &u_i0, &u_i1, &pebs, npoints, h, end_time);
    gettimeofday(&cpu_end, NULL);

    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6) - (cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    printf("CPU took %f seconds\n", elapsed_cpu);

    print_heatmap("lake_f.dat", u_cpu, npoints, h);

    free(u_cpu);

    mclose();
    bclose();
    remove(BACK_FILE);
    remove(MMAP_FILE);

    return 0;
}

void run_cpu(double* u, double* u0, double* u1, double* pebbles, int n, double h, double end_time) {
    double *un, *uc, *uo, *ut;

    un = (double*)malloc(sizeof(double) * n * n);
    uc = (double*)malloc(sizeof(double) * n * n);
    uo = (double*)malloc(sizeof(double) * n * n);

    memcpy(uo, u0, sizeof(double) * n * n);
    memcpy(uc, u1, sizeof(double) * n * n);

    int iteration_progress = 0;
    while (1) {
        evolve(un, uc, uo, pebbles, n, h, dt, t);

        ut = uo;
        uo = uc;
        uc = un;

        if (!tpdt(&t, dt, end_time))
            break;
        un = ut;

        iteration_progress++;
        if (iteration_progress < iteration_size)
            continue;

        checkpoint++;
        memcpy(u0, uo, sizeof(double) * n * n);
        memcpy(u1, uc, sizeof(double) * n * n);
        backup();
        iteration_progress = 0;
        printf("checkpoint %d passed\n", checkpoint);
    }

    memcpy(u, un, sizeof(double) * n * n);
}

void init_pebbles(double* p, int pn, int n) {
    int i, j, k, idx;
    int sz;

    // srand(time(NULL));
    srand(1);
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

void evolve(double* un, double* uc, double* uo, double* pebbles, int n, double h, double dt, double t) {
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
}

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
