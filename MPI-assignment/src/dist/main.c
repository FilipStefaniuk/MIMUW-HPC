#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "../utilities.h"
#include "../particle.h"
#include "simulation.h"

int main(int argc, char *argv[]) {

    int count;
    struct particle *particles = NULL;
    struct cmd_args cmd_args;
    struct mpi_process_data mpi_data;

    MPI_Init(&argc, &argv);

    parse_args(argc, argv, &cmd_args);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_data.proc_nr);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_data.proc_count);

    if (mpi_data.proc_nr == ROOT_PROCESS) {
        particles = parse_input(cmd_args.particles_in, &count);
    }

    MPI_Bcast(&count, 1, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);
    
    run_simulation(particles, count, &mpi_data, &cmd_args);

    if (mpi_data.proc_nr == ROOT_PROCESS) {
        free(particles);
    }

    MPI_Finalize();
    
    return 0;
}
