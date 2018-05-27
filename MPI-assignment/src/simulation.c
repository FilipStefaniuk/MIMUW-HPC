#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "particle.h"
#include "simulation.h"
#include "utilities.h"

//-------------------------------------------------------------------------------------------------
//                                 INIT / CLEANUP SIMULATION
//-------------------------------------------------------------------------------------------------

static void init_particle_type(MPI_Datatype *type) {
    
    int blocklengths[1] = {12};
    MPI_Datatype types[1] = {MPI_DOUBLE};
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

    MPI_Type_create_struct(1, blocklengths, offsets, types, type);
    MPI_Type_commit(type);
}


static void init_particles_data(struct particles_data *p_data, int count, 
                                            const struct mpi_process_data *mpi_data) {
    
    int displ = 0, split, rest;
    split = count / mpi_data->proc_count;
    rest = count % mpi_data->proc_count;

    p_data->p_count = count;

    p_data->sendcounts = malloc(sizeof(int) * mpi_data->proc_count);
    p_data->displs = malloc(sizeof(int) * mpi_data->proc_count);
    
    for (int i = 0; i < mpi_data->proc_count; ++i) {

        p_data->displs[i] = displ;
        p_data->sendcounts[i] = split + (rest > i);

        displ += p_data->sendcounts[i];
    }

    p_data->my_count = split + (rest > mpi_data->proc_nr);
    p_data->up_count = split + (rest != 0);

    init_particle_type(&p_data->type);
}

static void free_particles_data(struct particles_data *p_data) {
    
    free(p_data->sendcounts);
    free(p_data->displs);
}

static void alloc_particle_buffer(struct particles_buff *b, size_t size) {
    b->buff = calloc(size, sizeof(struct particle));
    b->buff_rcv = calloc(size, sizeof(struct particle));
}

static void free_particle_buffer(struct particles_buff *b) {
    free(b->buff);
    free(b->buff_rcv);
}

//-------------------------------------------------------------------------------------------------
//                                3 BODY ALGORITHM
//-------------------------------------------------------------------------------------------------

static void exchange_buffers(struct particles_buff *b, 
                             struct particles_data *p_data, 
                             struct mpi_process_data *mpi_data) {
                                     MPI_Request requests[4];
    MPI_Status statuses[4];

    b[PREV_BUFF(MY_BUFF)].owner = PREV_PROC(mpi_data->proc_nr, mpi_data->proc_count);
    b[NEXT_BUFF(MY_BUFF)].owner = NEXT_PROC(mpi_data->proc_nr, mpi_data->proc_count);

    b[PREV_BUFF(MY_BUFF)].count = p_data->sendcounts[b[PREV_BUFF(MY_BUFF)].owner];
    b[NEXT_BUFF(MY_BUFF)].count = p_data->sendcounts[b[NEXT_BUFF(MY_BUFF)].owner];

    MPI_Irecv(b[PREV_BUFF(MY_BUFF)].buff, b[PREV_BUFF(MY_BUFF)].count, p_data->type,
                b[PREV_BUFF(MY_BUFF)].owner, 1, MPI_COMM_WORLD, &requests[0]);

    MPI_Irecv(b[NEXT_BUFF(MY_BUFF)].buff, b[NEXT_BUFF(MY_BUFF)].count, p_data->type,
        b[NEXT_BUFF(MY_BUFF)].owner, 1, MPI_COMM_WORLD, &requests[1]);

    MPI_Isend(b[MY_BUFF].buff, b[MY_BUFF].count, p_data->type, 
              PREV_PROC(mpi_data->proc_nr, mpi_data->proc_count), 1, MPI_COMM_WORLD, &requests[2]);

    MPI_Isend(b[MY_BUFF].buff, b[MY_BUFF].count, p_data->type, 
              NEXT_PROC(mpi_data->proc_nr, mpi_data->proc_count), 1, MPI_COMM_WORLD, &requests[3]);

    MPI_Waitall(4, requests, statuses);

}

static void group_buffers(struct particles_buff *b, 
                          struct particles_data *p_data, 
                          struct mpi_process_data *mpi_data) {

    int request_count = 0;
    MPI_Request requests2[NUM_BUFFS * 2];
    MPI_Status statuses2[NUM_BUFFS * 2];

    for (int i = 0; i < NUM_BUFFS; ++i) {
        if (b[i].owner != mpi_data->proc_nr) {

            MPI_Irecv(b[i].buff_rcv, p_data->my_count, p_data->type,
                MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &requests2[request_count]);

            request_count++;

            MPI_Isend(b[i].buff, b[i].count, p_data->type, b[i].owner, 1,
                MPI_COMM_WORLD, &requests2[request_count]);
            
            request_count++;
        }
    }

    MPI_Waitall(request_count, requests2, statuses2);

    for (int i = 0; i < NUM_BUFFS; ++i) {

        if (b[i].owner != mpi_data->proc_nr) {

            memcpy(b[i].buff, b[i].buff_rcv, p_data->my_count * sizeof(struct particle));
            b[i].owner = mpi_data->proc_nr;
            b[i].count = p_data->my_count;
        }
    }
}

static void shift_right(struct particles_buff *b, 
                        struct particles_data *p_data, struct mpi_process_data *mpi_data) {

    int prev_owner, prev_count;
    MPI_Request requests[2];
    MPI_Status statuses[2];

    prev_owner = PREV_PROC(b->owner, mpi_data->proc_count);
    prev_count = p_data->sendcounts[prev_owner];

    MPI_Irecv(b->buff_rcv, prev_count, p_data->type, 
        PREV_PROC(mpi_data->proc_nr, mpi_data->proc_count),1, MPI_COMM_WORLD, &requests[0]);

    MPI_Isend(b->buff, b->count, p_data->type,
        NEXT_PROC(mpi_data->proc_nr, mpi_data->proc_count), 1, MPI_COMM_WORLD, &requests[1]);

    MPI_Waitall(2, requests, statuses);

    memcpy(b->buff, b->buff_rcv, prev_count * sizeof(struct particle));
    b->owner = prev_owner;
    b->count = prev_count;
}

void one_buff_interactions(struct particles_buff *b) {
    for (int i = 0; i < b->count; ++i) {
        for (int j = i + 1; j < b->count; ++j) {
            for (int k = j + 1; k < b->count; ++k) {
                compute_force(&b->buff[i], &b->buff[j], &b->buff[k]);
                compute_force(&b->buff[j], &b->buff[i], &b->buff[k]);
                compute_force(&b->buff[k], &b->buff[j], &b->buff[i]);
            }
        }
    }
}

void two_buff_interactions(struct particles_buff *b0, struct particles_buff *b1) {
    for (int i = 0; i < b0->count; ++i) {
        for (int j = i + 1; j < b0->count; ++j) {
            for (int k = 0; k < b1->count; ++k) {
                compute_force(&b0->buff[i], &b0->buff[j], &b1->buff[k]);
                compute_force(&b0->buff[j], &b0->buff[i], &b1->buff[k]);
                compute_force(&b1->buff[k], &b0->buff[j], &b0->buff[i]);
            }
        }
    }
}

void three_buff_interactions(struct particles_buff *b0, struct particles_buff *b1, struct particles_buff *b2) {
    for (int i = 0; i < b0->count; ++i) {
        for (int j = 0; j < b1->count; ++j) {
            for (int k = 0; k < b2->count; ++k) {
                compute_force(&b0->buff[i], &b1->buff[j], &b2->buff[k]);
                compute_force(&b1->buff[j], &b0->buff[i], &b2->buff[k]);
                compute_force(&b2->buff[k], &b1->buff[j], &b0->buff[i]);
            }
        }
    }
}

void one_third_interactions(struct particles_buff *b0, struct particles_buff *b1, struct particles_buff *b2) {
    for (int i = 0; i < b0->count; ++i) {
        for (int j = 0; j < b1->count; ++j) {
            for (int k = 0; k < b2->count; ++k) {
                compute_force(&b0->buff[i], &b1->buff[j], &b2->buff[k]);
            }
        }
    }
}

static void compute_accelerations(struct particles_buff *b,
                                  struct particles_data *p_data, struct mpi_process_data *mpi_data) {

    int shift = 0;
    exchange_buffers(b, p_data, mpi_data);

    for (int s = mpi_data->proc_count - 3; s >= 0; s -= 3) {

        for (int i = 0; i < s; ++i) {
            
            if (i != 0 || s != mpi_data->proc_count - 3) {
            
                shift_right(&b[shift], p_data, mpi_data);

            } else {
            
                one_buff_interactions(&b[MY_BUFF]);
                two_buff_interactions(&b[MY_BUFF], &b[NEXT_BUFF(MY_BUFF)]);
                two_buff_interactions(&b[PREV_BUFF(MY_BUFF)], &b[NEXT_BUFF(MY_BUFF)]);
            }

            if (s == mpi_data->proc_count - 3) {
                two_buff_interactions(&b[MY_BUFF], &b[PREV_BUFF(MY_BUFF)]);
            }

            three_buff_interactions(&b[PREV_BUFF(MY_BUFF)], &b[MY_BUFF], &b[NEXT_BUFF(MY_BUFF)]);
        }

        shift = NEXT_BUFF(shift);
    }

    if (mpi_data->proc_count % 3 == 0) {

        shift_right(&b[PREV_BUFF(shift)], p_data, mpi_data);
        one_third_interactions(&b[PREV_BUFF(MY_BUFF)], &b[MY_BUFF], &b[NEXT_BUFF(MY_BUFF)]);
    }



    group_buffers(b, p_data, mpi_data);

    for (int i = 0; i < b[MY_BUFF].count; ++i) {

        b[MY_BUFF].buff[i].fx += b[PREV_BUFF(MY_BUFF)].buff[i].fx + b[NEXT_BUFF(MY_BUFF)].buff[i].fx;
        b[MY_BUFF].buff[i].fy += b[PREV_BUFF(MY_BUFF)].buff[i].fy + b[NEXT_BUFF(MY_BUFF)].buff[i].fy;
        b[MY_BUFF].buff[i].fz += b[PREV_BUFF(MY_BUFF)].buff[i].fz + b[NEXT_BUFF(MY_BUFF)].buff[i].fz;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

//-------------------------------------------------------------------------------------------------
//                                    UPDATES
//-------------------------------------------------------------------------------------------------

static void update_positions(struct particles_buff *b, double deltatime) {
    for (int i = 0; i < b->count; ++i) {
        update_position(&b->buff[i], deltatime);
    }
}

static void update_velocities(struct particles_buff *b, double deltatime) {
    for (int i = 0; i < b->count; ++i) {
        update_velocity(&b->buff[i], deltatime);
    }
}

static void update_accelerations(struct particles_buff *b) {
    for (int i = 0; i < b->count; ++i) {
        update_acceleration(&b->buff[i]);
    }
}

//-------------------------------------------------------------------------------------------------
//                                     SIMULATION
//-------------------------------------------------------------------------------------------------

void run_simulation(struct particle *particles, int count,
                    struct mpi_process_data *mpi_data, struct cmd_args *cmd_args) {

    struct particles_data p_data;
    struct particles_buff b[NUM_BUFFS];

    // Initialization
    init_particles_data(&p_data, count, mpi_data);
    for (int i = 0; i < NUM_BUFFS; ++i) {
        alloc_particle_buffer(&b[i], p_data.up_count);
    }
    
    // Scatter particles between nodes
    b[MY_BUFF].count = p_data.my_count;
    b[MY_BUFF].owner = mpi_data->proc_nr;
    MPI_Scatterv(particles, p_data.sendcounts, p_data.displs, p_data.type, 
            b[MY_BUFF].buff, b[MY_BUFF].count, p_data.type, ROOT_PROCESS, MPI_COMM_WORLD);

    // Compute initial acceleration
    compute_accelerations(b, &p_data, mpi_data);
    update_accelerations(&b[MY_BUFF]);

    // Run simulation steps
    for (int i = 0; i < cmd_args->stepcount; ++i) {

        update_positions(&b[MY_BUFF], cmd_args->deltatime);
        compute_accelerations(b, &p_data, mpi_data);
        update_velocities(&b[MY_BUFF], cmd_args->deltatime);
        update_accelerations(&b[MY_BUFF]);

        if (cmd_args->verbose || i + 1 == cmd_args->stepcount) {
        
            MPI_Gatherv(b[MY_BUFF].buff, b[MY_BUFF].count, p_data.type, particles,
                p_data.sendcounts, p_data.displs, p_data.type, ROOT_PROCESS, MPI_COMM_WORLD);

            if (mpi_data->proc_nr == ROOT_PROCESS && cmd_args->verbose) {
                write_log(cmd_args->particles_out, p_data.p_count, particles, i + 1);
            }
        }
    }

    // Write output
    if (mpi_data->proc_nr == ROOT_PROCESS) {
        write_output(cmd_args->particles_out, p_data.p_count, particles);
    }
    
    // Cleanup
    free_particles_data(&p_data);
    for (int i = 0; i < NUM_BUFFS; ++i) {
        free_particle_buffer(&b[i]);
    }
}