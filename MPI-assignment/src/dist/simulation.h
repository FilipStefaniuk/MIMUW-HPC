#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include <mpi.h>
#include "../utilities.h"

#define ROOT_PROCESS 0
#define NUM_BUFFS 3
#define MY_BUFF 1

#define NEXT_PROC(NR, COUNT) (NR + 1) % COUNT
#define PREV_PROC(NR, COUNT) (COUNT + NR - 1) % COUNT
#define NEXT_BUFF(NR) (NR + 1) % NUM_BUFFS
#define PREV_BUFF(NR) (NUM_BUFFS + NR - 1) % NUM_BUFFS

struct particles_data {
    int p_count;
    int my_count;
    int up_count;
    int *sendcounts;
    int *displs;
    MPI_Datatype type;
};

struct particles_buff {
    int owner;
    int count;
    struct particle *buff;
    struct particle *buff_rcv;
};

struct mpi_process_data {
    int proc_nr;
    int proc_count;
};

void run_simulation(struct particle *particles, int count, 
                    struct mpi_process_data *mpi_data, struct cmd_args *cmd_args);

#endif
