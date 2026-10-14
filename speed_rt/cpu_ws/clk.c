/* clk N: the effective clock (MHz) of the vCPU this process runs on: a chain of 4 dependent adds per iteration (one
   cycle each on x86-64), N iterations, timed with CLOCK_MONOTONIC. harness.py runs it pinned to each slot right
   after each reference segment. Build: gcc -O2 -o clk clk.c */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 20000000L;
    uint64_t x = 1;
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < n; i++)
        __asm__ volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
    clock_gettime(CLOCK_MONOTONIC, &b);
    double s = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
    printf("%.1f\n", 4.0 * n / s / 1e6);
    return x == 0;
}
