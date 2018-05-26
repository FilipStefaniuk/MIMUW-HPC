#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

static int const ROOT_PROCESS = 0;
static int const NUM_PER_PROCESS = 2;


// void parse_input(char *filename, )

int main(int argc, char *argv[]) {
    
    int numProcesses, myProcessNo;
    int *allNumbers, *myNumbers;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myProcessNo);

    myNumbers = (int*)malloc(sizeof(int) * NUM_PER_PROCESS);


    if (myProcessNo == ROOT_PROCESS) {

        allNumbers = (int *)malloc(sizeof(int)*NUM_PER_PROCESS*numProcesses);

        for (int i = 0; i < NUM_PER_PROCESS * numProcesses; ++i) {
            allNumbers[i] = rand() % (NUM_PER_PROCESS * numProcesses);
            printf("%d\n", allNumbers[i]);
        }
        printf("--\n");
    }

    MPI_Scatter(
        allNumbers,
        NUM_PER_PROCESS,
        MPI_INT,
        myNumbers,
        NUM_PER_PROCESS,
        MPI_INT,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < NUM_PER_PROCESS; ++i) {
        myNumbers[i] *= myNumbers[i];
    }

    MPI_Gather(
        myNumbers,
        NUM_PER_PROCESS,
        MPI_INT,
        allNumbers,
        NUM_PER_PROCESS,
        MPI_INT,
        ROOT_PROCESS,
        MPI_COMM_WORLD
    );

    if (myProcessNo == ROOT_PROCESS) {
        for (int i = 0; i < NUM_PER_PROCESS * numProcesses; ++ i) {
            printf("%d\n", allNumbers[i]);
        }

        free(allNumbers);
    }
    free(myNumbers);
    MPI_Finalize();

    return 0;
}