#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include "particle.h"

struct cmd_args {
    char *particles_in;
    char *particles_out;
    int stepcount;
    double deltatime;
    int verbose;
};

int parse_args(int argc, char *argv[], struct cmd_args *cmd_args);

struct particle * parse_input(char *filename, int *count);

int write_output(char *filename, int count, struct particle *p);

#endif