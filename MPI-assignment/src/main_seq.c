#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utilities.h"
#include "particle.h"

int main(int argc, char *argv[]) {

    int allParticlesCount;
    struct particle *allParticles = NULL;
    struct cmd_args cmd_args;

    parse_args(argc, argv, &cmd_args);

    allParticles = parse_input(cmd_args.particles_in, &allParticlesCount);

    // Compute initial acceleration
    for (int i = 0; i < allParticlesCount; ++i) {
        for (int j = i + 1;  j < allParticlesCount; ++j) {
            for (int k = j + 1; k < allParticlesCount; ++k) {
                // printf("%d %d %d\n", i, j, k);

                double tmp1, tmp2, tmp3;
               
                tmp1 = allParticles[i].fx;
                tmp2 = allParticles[j].fx;
                tmp3 = allParticles[k].fx;

                allParticles[i].fx = 0;
                allParticles[j].fx = 0;
                allParticles[k].fx = 0;

                compute_force(&allParticles[i], &allParticles[j], &allParticles[k]);
                compute_force(&allParticles[j], &allParticles[i], &allParticles[k]);
                compute_force(&allParticles[k], &allParticles[j], &allParticles[i]);
                
                // printf("{%d} %d %d: x: %lf %lf %lf | %.16lf\n", i, j, k, allParticles[i].x, allParticles[j].x, allParticles[k].x, allParticles[i].fx);
                // printf("{%d} %d %d: x: %lf %lf %lf | %.16lf\n", j, i, k, allParticles[j].x, allParticles[i].x, allParticles[k].x, allParticles[j].fx);
                // printf("{%d} %d %d: x: %lf %lf %lf | %.16lf\n", k, j, i, allParticles[k].x, allParticles[j].x, allParticles[i].x, allParticles[k].fx);
            
                allParticles[i].fx += tmp1;
                allParticles[j].fx += tmp2;
                allParticles[k].fx += tmp3;
                // exit(0);
            }
        }

        printf("New accs: %.15lf, %.15lf, %.15lf\n", allParticles[i].fx, allParticles[i].fy, allParticles[i].fz);

        update_acceleration(&allParticles[i]);
    }

    // Compute in loop
    for (int step = 0; step < cmd_args.stepcount; ++step) {

        for (int i = 0; i < allParticlesCount; ++i) {
            update_position(&allParticles[i], cmd_args.deltatime);
        }

        for (int i = 0; i < allParticlesCount; ++i) {
            for (int j = i + 1;  j < allParticlesCount; ++j) {
                for (int k = j + 1; k < allParticlesCount; ++k) {
                    compute_force(&allParticles[i], &allParticles[j], &allParticles[k]);
                    compute_force(&allParticles[j], &allParticles[i], &allParticles[k]);
                    compute_force(&allParticles[k], &allParticles[j], &allParticles[i]);
                }
            }

            printf("New accs: %.15lf, %.15lf, %.15lf\n", allParticles[i].fx, allParticles[i].fy, allParticles[i].fz);

            update_velocity(&allParticles[i], cmd_args.deltatime);
            update_acceleration(&allParticles[i]);
        }
    }

    write_output(cmd_args.particles_out, allParticlesCount, allParticles);

    if (allParticles != NULL) {
        free(allParticles);
    }
}