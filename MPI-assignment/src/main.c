#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utilities.h"

void shift_right(int *owner, int *count, struct particle *b, int prev, int *sendcounts, int numProcesses, int next, int particlesCountRoundUp, MPI_Datatype *datatype) {

    MPI_Request requests[2];
    MPI_Status statuses[2];

    int prev_owner = (numProcesses + *owner - 1) % numProcesses;
    int prev_count = sendcounts[prev_owner];

    struct particle *buff = malloc(particlesCountRoundUp * sizeof(struct particle));

    MPI_Irecv(
        buff,
        prev_count,
        *datatype,
        prev,
        1,
        MPI_COMM_WORLD,
        &requests[0]
    );

    MPI_Isend(
        b,
        *count,
        *datatype,
        next,
        1,
        MPI_COMM_WORLD,
        &requests[1]
    );

    MPI_Waitall(2, requests, statuses);

    memcpy(b, buff, prev_count * sizeof(struct particle));
    *owner = prev_owner;
    *count = prev_count;

    free(buff);
}

void calculate_one_buffer(int count, struct particle *b) {
    
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            for (int k = j + 1; k < count; ++k) {
                // printf("One buffer calculation!\n");
                compute_force(&b[i], &b[j], &b[k]);
                compute_force(&b[j], &b[i], &b[k]);
                compute_force(&b[k], &b[j], &b[i]);
            }
        }
    }
}

void calculate_two_buffer(int num, int count0, struct particle *b0, int count1, struct particle *b1) {

    for (int i = 0; i < count0; ++i) {
        for (int j = i + 1; j < count0; ++j) {
            for (int k = 0; k < count1; ++k) {
                double tmp1, tmp2, tmp3;

                tmp1 = b0[i].fx;
                tmp2 = b0[j].fx;
                tmp3 = b1[k].fx;

                b0[i].fx = 0;
                b0[j].fx = 0;
                b1[k].fx = 0;

                // printf("Two buffer calculation!\n");
                compute_force(&b0[i], &b0[j], &b1[k]);
                compute_force(&b0[j], &b0[i], &b1[k]);
                compute_force(&b1[k], &b0[j], &b0[i]);

                // printf("{%d} x: %lf %lf %lf %.16lf\n", num, b0[i].x, b0[j].x, b1[k].x,  b0[i].fx);
                // printf("{%d} x: %lf %lf %lf %.16lf\n", num, b0[j].x, b0[i].x, b1[k].x,  b0[j].fx);
                // printf("{%d} x: %lf %lf %lf %.16lf\n", num, b1[k].x, b0[j].x, b0[i].x,  b1[k].fx);

                b0[i].fx += tmp1;
                b0[j].fx += tmp2;
                b1[k].fx += tmp3;
            }
        }
    }
}

void calculate_three_buffer(int num, int count0, struct particle *b0, int count1, struct particle *b1, int count2, struct particle *b2, int owner0, int owner1, int owner2) {
    
    // int prev, next;
    // prev = (num + 3) % 4;
    // next = (num + 1) % 4;

    for (int i = 0; i < count0; ++i) {
        for (int j = 0; j < count1; ++j) {
            for (int k = 0; k < count2; ++k) {
                double tmp1, tmp2, tmp3;

                tmp1 = b0[i].fx;
                tmp2 = b1[j].fx;
                tmp3 = b2[k].fx;

                b0[i].fx = 0;
                b1[j].fx = 0;
                b2[k].fx = 0;

                // printf("{%d} (%d/%d, {%d}) (%d/%d, {%d}) (%d/%d, {%d})\n", num, i+1, count0, owner0, j+1, count1, owner1, k+1, count2, owner2);

                compute_force(&b0[i], &b1[j], &b2[k]);
                compute_force(&b1[j], &b0[i], &b2[k]);
                compute_force(&b2[k], &b1[j], &b0[i]);

                // printf("{%d} x: %lf %lf %lf %.16lf\n", num, b0[i].x, b1[j].x, b2[k].x,  b0[i].fx);
                // printf("{%d} x: %lf %lf %lf %.16lf\n", num, b1[j].x, b0[i].x, b2[k].x,  b1[j].fx);
                // printf("{%d} x: %lf %lf %lf %.16lf\n", num, b2[k].x, b1[j].x, b0[i].x,  b2[k].fx);

                b0[i].fx += tmp1;
                b1[j].fx += tmp2;
                b2[k].fx += tmp3;
            }
        }
    }
}

void calculate_one_third(int num, int count0, struct particle *b0, int count1, struct particle *b1, int count2, struct particle *b2, int owner0, int owner1, int owner2) {
    
    for (int i = 0; i < count0; ++i) {
        for (int j = 0; j < count1; ++j) {
            for (int k = 0; k < count2; ++k) {
                double tmp1;

                tmp1 = b0[i].fx;

                b0[i].fx = 0;

                // printf("{%d} (%d/%d, {%d}) (%d/%d, {%d}) (%d/%d, {%d})\n", num, i+1, count0, owner0, j+1, count1, owner1, k+1, count2, owner2);

                compute_force(&b0[i], &b1[j], &b2[k]);

                // printf("{%d} x: %lf %lf %lf %.16lf\n", num, b0[i].x, b1[j].x, b2[k].x,  b0[i].fx);

                b0[i].fx += tmp1;
            }
        }
    }
}


static int const ROOT_PROCESS = 0;

int main(int argc, char *argv[]) {

    int allParticlesCount = 0, particlesCountRoundUp;
    int *sendcounts, *displs;
    struct particle *allParticles = NULL;
    int b_owner[3];
    int b_count[3];
    struct particle * b[3];
    int numProcesses, myProcessNo, myParticlesCount;
    int next, prev;
    int shift_next;
    double deltatime = 0.5;
    struct cmd_args cmd_args;

    parse_args(argc, argv, &cmd_args);

    MPI_Init(&argc, &argv);

    const int nitems = 1;
    int blocklengths[1] = {12};
    MPI_Datatype types[1] = {MPI_DOUBLE};
    MPI_Datatype mpi_particle_type;
    MPI_Aint offsets[12] = {
        offsetof(struct particle, x),
        offsetof(struct particle, y),       
        offsetof(struct particle, z),
        offsetof(struct particle, vx),
        offsetof(struct particle, vy),
        offsetof(struct particle, vz),
        offsetof(struct particle, ax),
        offsetof(struct particle, ay),
        offsetof(struct particle, az),
        offsetof(struct particle, fx),
        offsetof(struct particle, fy),
        offsetof(struct particle, fz)
    };

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_particle_type);
    MPI_Type_commit(&mpi_particle_type);

    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myProcessNo);

    if (myProcessNo == ROOT_PROCESS) {
        allParticles = parse_input(cmd_args.particles_in, &allParticlesCount);

    }

    prev = (myProcessNo + numProcesses - 1) % numProcesses;
    next = (myProcessNo + 1) % numProcesses;


    MPI_Bcast(&allParticlesCount, 1, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);

    sendcounts = malloc(sizeof(int) * numProcesses);
    displs = malloc(sizeof(int) * numProcesses);

    // Compute n of particles for each process
    for (int i = 0, displ = 0; i < numProcesses; ++i) {
        sendcounts[i] = allParticlesCount / numProcesses + (allParticlesCount % numProcesses > i);
        displs[i] = displ;
        displ += sendcounts[i];
    }

    // Allocate rounded up buffers
    particlesCountRoundUp = allParticlesCount / numProcesses + (allParticlesCount % numProcesses != 0);
    b[0] = calloc(particlesCountRoundUp, sizeof(struct particle));
    b[1] = calloc(particlesCountRoundUp, sizeof(struct particle));
    b[2] = calloc(particlesCountRoundUp, sizeof(struct particle));

    // My particles
    myParticlesCount = allParticlesCount / numProcesses + (allParticlesCount % numProcesses > myProcessNo);
    b_owner[1] = myProcessNo;
    b_count[1] = myParticlesCount;

    MPI_Scatterv(
        allParticles,
        sendcounts,
        displs,
        mpi_particle_type,
        b[1],
        b_count[1],
        mpi_particle_type,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );

    // Algorithm
    //-------------------------------------------------------------------------

    MPI_Request requests[4];
    MPI_Status statuses[4];

    b_owner[0] = prev;
    b_owner[2] = next;

    b_count[0] = sendcounts[b_owner[0]];
    b_count[2] = sendcounts[b_owner[2]];

    MPI_Irecv(
        b[0],
        b_count[0],
        mpi_particle_type,
        b_owner[0],
        1,
        MPI_COMM_WORLD,
        &requests[0]
    );

    MPI_Irecv(
        b[2],
        b_count[2],
        mpi_particle_type,
        b_owner[2],
        1,
        MPI_COMM_WORLD,
        &requests[1]
    );

    MPI_Isend(
        b[1],
        b_count[1],
        mpi_particle_type,
        b_owner[0],
        1,
        MPI_COMM_WORLD,
        &requests[2]
    );

    MPI_Isend(
        b[1],
        b_count[1],
        mpi_particle_type,
        b_owner[2],
        1,
        MPI_COMM_WORLD,
        &requests[3]
    );

    MPI_Waitall(4, requests, statuses);


    shift_next = 0;
    // printf("%d: %d %d %d\n", myProcessNo, b_owner[0], b_owner[1], b_owner[2]);

    for (int s = numProcesses - 3; s >= 0; s -= 3) {

        for (int i = 0; i < s; ++i) {
            
            if (i != 0 || s != numProcesses - 3) {
            
                shift_right(&b_owner[shift_next], &b_count[shift_next], b[shift_next], prev,
                            sendcounts, numProcesses,  next, particlesCountRoundUp, &mpi_particle_type);

            } else {
            
                calculate_one_buffer(b_count[1], b[1]);
                calculate_two_buffer(myProcessNo, b_count[1], b[1], b_count[2], b[2]);
                calculate_two_buffer(myProcessNo, b_count[0], b[0], b_count[2], b[2]);
            }

            if (s == numProcesses - 3) {
                calculate_two_buffer(myProcessNo, b_count[1], b[1], b_count[0], b[0]);
            }

            // printf("%d: %d (%d) %d (%d) %d (%d)\n", myProcessNo, b_owner[0], b_count[0], b_owner[1], b_count[1], b_owner[2], b_count[2]);
            calculate_three_buffer(myProcessNo, b_count[0], b[0], b_count[1], b[1], b_count[2], b[2], b_owner[0], b_owner[1], b_owner[2]);

        }

        shift_next = (shift_next + 1) % 3;
    }

    // SPECIAL CASE
    if (numProcesses % 3 == 0) {


        shift_right(&b_owner[(shift_next + 2) % 3], &b_count[(shift_next + 2) % 3], b[(shift_next + 2) % 3], 
                    prev, sendcounts, numProcesses,  next, particlesCountRoundUp, &mpi_particle_type);

        calculate_one_third(myProcessNo, b_count[0], b[0], b_count[1], b[1], b_count[2], b[2], b_owner[0], b_owner[1], b_owner[2]);
    }

    // printf("< {%d} %.16lf %.16lf %.16lf\n", myProcessNo, b[0][0].x, b[1][0].x, b[2][0].x);


    // Sent particles back to owners
    {
        struct particle *buff[3];
        int request_count = 0;
        MPI_Request requests2[6];
        MPI_Status statuses2[6];

        buff[0] = malloc(sizeof(struct particle) * particlesCountRoundUp);
        buff[1] = malloc(sizeof(struct particle) * particlesCountRoundUp);
        buff[2] = malloc(sizeof(struct particle) * particlesCountRoundUp);

        for (int i = 0; i < 3; ++i) {
            if (b_owner[i] != myProcessNo) {

                // printf("<< {%d} %lf %.14lf\n", b_owner[i], b[i][0].x, b[i][0].fx);
                
                MPI_Irecv(
                    buff[i],
                    myParticlesCount,
                    mpi_particle_type,
                    MPI_ANY_SOURCE,
                    1,
                    MPI_COMM_WORLD,
                    &requests2[request_count]
                );

                request_count++;

                MPI_Isend(
                    b[i],
                    b_count[i],
                    mpi_particle_type,
                    b_owner[i],
                    1,
                    MPI_COMM_WORLD,
                    &requests2[request_count]
                );

                request_count++;
            }
        }

        MPI_Waitall(request_count, requests2, statuses2);

        // printf("%d: requests %d\n", myProcessNo, request_count);

        for (int i = 0; i < 3; ++i) {

            if (b_owner[i] != myProcessNo) {

                // printf(">> %d: %lf\n", myProcessNo, buff[i]->x);
                
                memcpy(b[i], buff[i], myParticlesCount * sizeof(struct particle));
                b_owner[i] = myProcessNo;
                b_count[i] = myParticlesCount;
            }
        }

        free(buff[0]);
        free(buff[1]);
        free(buff[2]);
    }

    // printf("> {%d} %.16lf %.16lf %.16lf\n", myProcessNo, b[0][0].x, b[1][0].x, b[2][0].x);

    // calculate (sum forces, change velocities)
    for (int i = 0; i < myParticlesCount; ++i) {

        // printf("{%d, %d} %.16lf %.16lf %.16lf\n", i, myProcessNo, b[0][i].fx, b[1][i].fx, b[2][i].fx);

        b[1][i].fx += b[0][i].fx + b[2][i].fx;
        b[1][i].fy += b[0][i].fy + b[2][i].fy;
        b[1][i].fz += b[0][i].fz + b[2][i].fz;

        printf("%d: New accs: %.15lf, %.15lf, %.15lf\n", myProcessNo, b[1][i].fx, b[1][i].fy, b[1][i].fz);

        update_acceleration(&b[1][i]);

        update_position(&b[1][i], deltatime);

        // printf("{%d} position %.16lf %.16lf %.16lf\n", myProcessNo, b[1][i].x, b[1][i].y, b[1][i].z);
    }

    //-------------------------------------------------------------------------


    MPI_Gatherv(
        b[1],
        b_count[1],
        mpi_particle_type,
        allParticles,
        sendcounts,
        displs,
        mpi_particle_type,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );

    if (myProcessNo == ROOT_PROCESS) {
        write_output(cmd_args.particles_out, allParticlesCount, allParticles);
    }

    free(b[0]);
    free(b[1]);
    free(b[2]);
    
    if (myProcessNo == ROOT_PROCESS) {
        free(displs);
        free(sendcounts);
        free(allParticles);
    }

    MPI_Finalize();

    return 0; 
}