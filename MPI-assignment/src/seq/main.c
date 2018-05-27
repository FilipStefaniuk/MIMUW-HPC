#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../utilities.h"
#include "../particle.h"

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

                compute_force(&allParticles[i], &allParticles[j], &allParticles[k]);
                compute_force(&allParticles[j], &allParticles[i], &allParticles[k]);
                compute_force(&allParticles[k], &allParticles[j], &allParticles[i]);
            }
        }

        // printf("New accs: %.15lf, %.15lf, %.15lf\n", allParticles[i].fx, allParticles[i].fy, allParticles[i].fz);

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

            // printf("New accs: %.15lf, %.15lf, %.15lf\n", allParticles[i].fx, allParticles[i].fy, allParticles[i].fz);

            update_velocity(&allParticles[i], cmd_args.deltatime);
            update_acceleration(&allParticles[i]);
        }

        if (cmd_args.verbose) {
            write_log(cmd_args.particles_out, allParticlesCount, allParticles, step+1);
        }
    }

    write_final_output(cmd_args.particles_out, allParticlesCount, allParticles);

    if (allParticles != NULL) {
        free(allParticles);
    }
}